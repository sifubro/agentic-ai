"""
Message queue implementations for distributed task execution.
Supports RabbitMQ and Redis Streams.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple,
    Awaitable, Union
)
import logging

from core.types import Message, MessageType, generate_id, current_timestamp

logger = logging.getLogger(__name__)


class MessageQueue(ABC):
    """Abstract base class for message queues."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the message queue."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message queue."""
        pass
    
    @abstractmethod
    async def publish(self, queue: str, message: Message) -> bool:
        """Publish a message to a queue."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        queue: str,
        callback: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe to a queue with a callback."""
        pass
    
    @abstractmethod
    async def acknowledge(self, queue: str, message_id: str) -> bool:
        """Acknowledge a message."""
        pass
    
    @abstractmethod
    async def reject(
        self,
        queue: str,
        message_id: str,
        requeue: bool = True
    ) -> bool:
        """Reject a message."""
        pass


class InMemoryQueue(MessageQueue):
    """
    In-memory message queue for testing and single-instance deployments.
    """
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._pending: Dict[str, Dict[str, Message]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def connect(self) -> bool:
        """Connect to the queue."""
        self._running = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from the queue."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
    
    async def publish(self, queue: str, message: Message) -> bool:
        """Publish a message."""
        if queue not in self._queues:
            self._queues[queue] = asyncio.Queue()
            self._pending[queue] = {}
        
        await self._queues[queue].put(message)
        logger.debug(f"Published message {message.id} to queue {queue}")
        return True
    
    async def subscribe(
        self,
        queue: str,
        callback: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe to a queue."""
        if queue not in self._queues:
            self._queues[queue] = asyncio.Queue()
            self._pending[queue] = {}
        
        if queue not in self._subscribers:
            self._subscribers[queue] = []
        
        self._subscribers[queue].append(callback)
        
        # Start consumer task
        task = asyncio.create_task(self._consumer_loop(queue))
        self._tasks.append(task)
    
    async def acknowledge(self, queue: str, message_id: str) -> bool:
        """Acknowledge a message."""
        if queue in self._pending and message_id in self._pending[queue]:
            del self._pending[queue][message_id]
            return True
        return False
    
    async def reject(
        self,
        queue: str,
        message_id: str,
        requeue: bool = True
    ) -> bool:
        """Reject a message."""
        if queue in self._pending and message_id in self._pending[queue]:
            message = self._pending[queue][message_id]
            del self._pending[queue][message_id]
            
            if requeue:
                await self.publish(queue, message)
            
            return True
        return False
    
    async def _consumer_loop(self, queue: str):
        """Consumer loop for a queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._queues[queue].get(),
                    timeout=1.0
                )
                
                # Store in pending
                self._pending[queue][message.id] = message
                
                # Call subscribers
                for callback in self._subscribers.get(queue, []):
                    try:
                        await callback(message)
                        await self.acknowledge(queue, message.id)
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")
                        await self.reject(queue, message.id, requeue=True)
                        
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")


class RabbitMQQueue(MessageQueue):
    """
    RabbitMQ message queue implementation using pika.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/"
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        
        self._connection = None
        self._channel = None
        self._consumers: Dict[str, str] = {}  # queue -> consumer_tag
    
    async def connect(self) -> bool:
        """Connect to RabbitMQ."""
        try:
            import pika
            from pika.adapters.asyncio_connection import AsyncioConnection
            
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials
            )
            
            # For async operations, we use blocking connection in executor
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None,
                lambda: pika.BlockingConnection(parameters)
            )
            self._channel = self._connection.channel()
            
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
            return True
            
        except ImportError:
            logger.warning("pika not installed, using mock mode")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._connection and self._connection.is_open:
            self._connection.close()
    
    async def publish(self, queue: str, message: Message) -> bool:
        """Publish a message to RabbitMQ."""
        if not self._channel:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            
            # Declare queue
            await loop.run_in_executor(
                None,
                lambda: self._channel.queue_declare(queue=queue, durable=True)
            )
            
            # Publish message
            import pika
            await loop.run_in_executor(
                None,
                lambda: self._channel.basic_publish(
                    exchange='',
                    routing_key=queue,
                    body=message.to_json(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        message_id=message.id,
                        timestamp=int(time.time()),
                        priority=message.priority
                    )
                )
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(
        self,
        queue: str,
        callback: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe to a RabbitMQ queue."""
        if not self._channel:
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            # Declare queue
            await loop.run_in_executor(
                None,
                lambda: self._channel.queue_declare(queue=queue, durable=True)
            )
            
            # Set prefetch
            await loop.run_in_executor(
                None,
                lambda: self._channel.basic_qos(prefetch_count=1)
            )
            
            # Create wrapper callback
            def on_message(ch, method, properties, body):
                try:
                    message = Message.from_dict(json.loads(body))
                    asyncio.create_task(callback(message))
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            # Start consuming
            consumer_tag = await loop.run_in_executor(
                None,
                lambda: self._channel.basic_consume(
                    queue=queue,
                    on_message_callback=on_message
                )
            )
            
            self._consumers[queue] = consumer_tag
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
    
    async def acknowledge(self, queue: str, message_id: str) -> bool:
        """Acknowledge is handled in the callback for RabbitMQ."""
        return True
    
    async def reject(
        self,
        queue: str,
        message_id: str,
        requeue: bool = True
    ) -> bool:
        """Reject is handled in the callback for RabbitMQ."""
        return True


class RedisStreamQueue(MessageQueue):
    """
    Redis Streams message queue implementation.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = None,
        db: int = 0,
        consumer_group: str = "agentic-workers",
        consumer_name: str = None
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or generate_id()[:8]
        
        self._redis = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True
            )
            
            # Test connection
            await self._redis.ping()
            self._running = True
            
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except ImportError:
            logger.warning("redis not installed, using mock mode")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        if self._redis:
            await self._redis.close()
    
    async def publish(self, queue: str, message: Message) -> bool:
        """Publish a message to Redis Stream."""
        if not self._redis:
            return False
        
        try:
            # Add to stream
            message_data = {
                "id": message.id,
                "type": message.type.value,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "payload": json.dumps(message.payload),
                "timestamp": message.timestamp,
                "priority": str(message.priority)
            }
            
            await self._redis.xadd(queue, message_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(
        self,
        queue: str,
        callback: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe to a Redis Stream."""
        if not self._redis:
            return
        
        try:
            # Create consumer group if not exists
            try:
                await self._redis.xgroup_create(
                    queue,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
            except Exception:
                pass  # Group might already exist
            
            # Start consumer task
            task = asyncio.create_task(
                self._consumer_loop(queue, callback)
            )
            self._tasks.append(task)
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
    
    async def _consumer_loop(
        self,
        queue: str,
        callback: Callable[[Message], Awaitable[None]]
    ):
        """Consumer loop for Redis Stream."""
        while self._running:
            try:
                # Read from stream
                messages = await self._redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {queue: '>'},
                    count=10,
                    block=1000
                )
                
                if not messages:
                    continue
                
                for stream, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            # Parse message
                            message = Message(
                                id=data['id'],
                                type=MessageType(data['type']),
                                sender_id=data['sender_id'],
                                receiver_id=data['receiver_id'],
                                payload=json.loads(data['payload']),
                                timestamp=data['timestamp'],
                                priority=int(data['priority'])
                            )
                            
                            # Process
                            await callback(message)
                            
                            # Acknowledge
                            await self._redis.xack(
                                queue,
                                self.consumer_group,
                                message_id
                            )
                            
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(1)
    
    async def acknowledge(self, queue: str, message_id: str) -> bool:
        """Acknowledge a message."""
        if not self._redis:
            return False
        
        try:
            await self._redis.xack(queue, self.consumer_group, message_id)
            return True
        except Exception:
            return False
    
    async def reject(
        self,
        queue: str,
        message_id: str,
        requeue: bool = True
    ) -> bool:
        """Reject a message."""
        # Redis streams don't have reject, messages stay pending
        return True
    
    async def get_pending_count(self, queue: str) -> int:
        """Get count of pending messages."""
        if not self._redis:
            return 0
        
        try:
            info = await self._redis.xpending(queue, self.consumer_group)
            return info['pending'] if info else 0
        except Exception:
            return 0
    
    async def claim_stale_messages(
        self,
        queue: str,
        min_idle_time_ms: int = 60000
    ) -> int:
        """Claim messages that have been pending too long."""
        if not self._redis:
            return 0
        
        try:
            # Get pending messages
            pending = await self._redis.xpending_range(
                queue,
                self.consumer_group,
                min='-',
                max='+',
                count=100
            )
            
            claimed = 0
            for msg in pending:
                if msg['time_since_delivered'] > min_idle_time_ms:
                    await self._redis.xclaim(
                        queue,
                        self.consumer_group,
                        self.consumer_name,
                        min_idle_time_ms,
                        [msg['message_id']]
                    )
                    claimed += 1
            
            return claimed
            
        except Exception as e:
            logger.error(f"Failed to claim messages: {e}")
            return 0


class MessageBroker:
    """
    High-level message broker that abstracts queue implementation.
    """
    
    def __init__(self, queue: MessageQueue):
        self.queue = queue
        self._handlers: Dict[str, List[Callable]] = {}
    
    async def start(self):
        """Start the broker."""
        await self.queue.connect()
    
    async def stop(self):
        """Stop the broker."""
        await self.queue.disconnect()
    
    def on(self, queue: str):
        """Decorator to register a message handler."""
        def decorator(handler: Callable[[Message], Awaitable[None]]):
            if queue not in self._handlers:
                self._handlers[queue] = []
            self._handlers[queue].append(handler)
            
            # Subscribe to queue
            asyncio.create_task(self.queue.subscribe(queue, self._dispatch(queue)))
            return handler
        return decorator
    
    def _dispatch(self, queue: str) -> Callable[[Message], Awaitable[None]]:
        """Create a dispatcher for a queue."""
        async def dispatcher(message: Message):
            for handler in self._handlers.get(queue, []):
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {queue}: {e}")
        return dispatcher
    
    async def send(self, queue: str, message: Message) -> bool:
        """Send a message."""
        return await self.queue.publish(queue, message)
    
    async def request(
        self,
        queue: str,
        message: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """Send a request and wait for response."""
        response_queue = f"response-{message.id}"
        response_future: asyncio.Future = asyncio.Future()
        
        async def response_handler(msg: Message):
            if msg.correlation_id == message.id:
                response_future.set_result(msg)
        
        # Subscribe to response queue
        await self.queue.subscribe(response_queue, response_handler)
        
        # Send request
        message.reply_to = response_queue
        await self.send(queue, message)
        
        # Wait for response
        try:
            return await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            return None