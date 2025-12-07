"""
Authentication module with PBKDF2 password hashing and token management.
Implements secure authentication, token refresh, and revocation.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from core.types import (
    AuthToken, User, AuthenticationError, AuthorizationError,
    generate_id, generate_token, current_timestamp, is_expired
)


class PBKDF2Hasher:
    """
    PBKDF2 password hasher implementation.
    Uses SHA-256 with configurable iterations for secure password hashing.
    """
    
    ALGORITHM = "pbkdf2_sha256"
    ITERATIONS = 600_000  # OWASP recommended minimum for PBKDF2-SHA256
    SALT_LENGTH = 32
    HASH_LENGTH = 32
    
    @classmethod
    def hash_password(cls, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash a password using PBKDF2-SHA256.
        
        Args:
            password: Plain text password
            salt: Optional salt bytes (generated if not provided)
            
        Returns:
            Tuple of (password_hash, salt) both as hex strings
        """
        if salt is None:
            salt = os.urandom(cls.SALT_LENGTH)
        elif isinstance(salt, str):
            salt = bytes.fromhex(salt)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            cls.ITERATIONS,
            dklen=cls.HASH_LENGTH
        )
        
        return password_hash.hex(), salt.hex()
    
    @classmethod
    def verify_password(cls, password: str, stored_hash: str, salt: str) -> bool:
        """
        Verify a password against stored hash.
        
        Args:
            password: Plain text password to verify
            stored_hash: Stored password hash (hex string)
            salt: Salt used for hashing (hex string)
            
        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = cls.hash_password(password, bytes.fromhex(salt))
        return hmac.compare_digest(computed_hash, stored_hash)


class TokenManager:
    """
    Manages authentication tokens including access tokens and refresh tokens.
    Implements token generation, validation, refresh, and revocation.
    """
    
    ACCESS_TOKEN_LIFETIME = timedelta(hours=1)
    REFRESH_TOKEN_LIFETIME = timedelta(days=30)
    
    def __init__(self):
        self._tokens: Dict[str, AuthToken] = {}
        self._refresh_tokens: Dict[str, str] = {}  # refresh_token -> access_token
        self._revoked_tokens: Set[str] = set()
        self._user_tokens: Dict[str, List[str]] = {}  # user_id -> [token_ids]
        self._lock = asyncio.Lock()
    
    async def generate_token_pair(
        self, 
        user_id: str, 
        scopes: List[str] = None
    ) -> AuthToken:
        """
        Generate an access token and refresh token pair.
        
        Args:
            user_id: User ID to associate with tokens
            scopes: Optional list of permission scopes
            
        Returns:
            AuthToken with both access and refresh tokens
        """
        async with self._lock:
            now = datetime.utcnow()
            
            access_token = generate_token(48)
            refresh_token = generate_token(64)
            
            token = AuthToken(
                token=access_token,
                user_id=user_id,
                created_at=now.isoformat(),
                expires_at=(now + self.ACCESS_TOKEN_LIFETIME).isoformat(),
                refresh_token=refresh_token,
                refresh_expires_at=(now + self.REFRESH_TOKEN_LIFETIME).isoformat(),
                scopes=scopes or [],
                is_revoked=False
            )
            
            self._tokens[access_token] = token
            self._refresh_tokens[refresh_token] = access_token
            
            if user_id not in self._user_tokens:
                self._user_tokens[user_id] = []
            self._user_tokens[user_id].append(access_token)
            
            return token
    
    async def validate_token(self, token: str) -> Optional[AuthToken]:
        """
        Validate an access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            AuthToken if valid, None otherwise
        """
        async with self._lock:
            if token in self._revoked_tokens:
                return None
            
            auth_token = self._tokens.get(token)
            if not auth_token:
                return None
            
            if is_expired(auth_token.expires_at):
                return None
            
            return auth_token
    
    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """
        Use a refresh token to get a new access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New AuthToken if refresh successful, None otherwise
        """
        async with self._lock:
            access_token = self._refresh_tokens.get(refresh_token)
            if not access_token:
                return None
            
            old_token = self._tokens.get(access_token)
            if not old_token:
                return None
            
            if old_token.refresh_expires_at and is_expired(old_token.refresh_expires_at):
                # Refresh token expired, revoke everything
                await self._revoke_token_internal(access_token)
                return None
            
            # Revoke old token
            self._revoked_tokens.add(access_token)
            del self._tokens[access_token]
            del self._refresh_tokens[refresh_token]
            
            # Generate new token pair
            now = datetime.utcnow()
            new_access_token = generate_token(48)
            new_refresh_token = generate_token(64)
            
            new_token = AuthToken(
                token=new_access_token,
                user_id=old_token.user_id,
                created_at=now.isoformat(),
                expires_at=(now + self.ACCESS_TOKEN_LIFETIME).isoformat(),
                refresh_token=new_refresh_token,
                refresh_expires_at=(now + self.REFRESH_TOKEN_LIFETIME).isoformat(),
                scopes=old_token.scopes,
                is_revoked=False
            )
            
            self._tokens[new_access_token] = new_token
            self._refresh_tokens[new_refresh_token] = new_access_token
            
            # Update user tokens
            if old_token.user_id in self._user_tokens:
                self._user_tokens[old_token.user_id] = [
                    t for t in self._user_tokens[old_token.user_id] 
                    if t != access_token
                ]
                self._user_tokens[old_token.user_id].append(new_access_token)
            
            return new_token
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke an access token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if revoked, False if not found
        """
        async with self._lock:
            return await self._revoke_token_internal(token)
    
    async def _revoke_token_internal(self, token: str) -> bool:
        """Internal token revocation without lock."""
        auth_token = self._tokens.get(token)
        if not auth_token:
            return False
        
        self._revoked_tokens.add(token)
        auth_token.is_revoked = True
        
        # Also revoke refresh token
        if auth_token.refresh_token:
            if auth_token.refresh_token in self._refresh_tokens:
                del self._refresh_tokens[auth_token.refresh_token]
        
        return True
    
    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        async with self._lock:
            tokens = self._user_tokens.get(user_id, [])
            count = 0
            for token in tokens:
                if await self._revoke_token_internal(token):
                    count += 1
            return count
    
    async def get_token_info(self, token: str) -> Optional[Dict]:
        """Get detailed information about a token."""
        auth_token = await self.validate_token(token)
        if not auth_token:
            return None
        return auth_token.to_dict()
    
    async def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage."""
        async with self._lock:
            expired = []
            for token, auth_token in self._tokens.items():
                if is_expired(auth_token.expires_at):
                    expired.append(token)
            
            for token in expired:
                auth_token = self._tokens[token]
                if auth_token.refresh_token in self._refresh_tokens:
                    del self._refresh_tokens[auth_token.refresh_token]
                del self._tokens[token]
                self._revoked_tokens.discard(token)
            
            return len(expired)


class AuthenticationService:
    """
    Main authentication service handling user management and authentication.
    """
    
    def __init__(self, storage=None):
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}  # username -> user_id
        self._email_index: Dict[str, str] = {}  # email -> user_id
        self.token_manager = TokenManager()
        self.storage = storage
        self._lock = asyncio.Lock()
    
    async def register_user(
        self,
        username: str,
        password: str,
        email: str,
        roles: List[str] = None
    ) -> User:
        """
        Register a new user.
        
        Args:
            username: Unique username
            password: Plain text password
            email: User email
            roles: Optional list of roles
            
        Returns:
            Created User object
            
        Raises:
            AuthenticationError: If username or email already exists
        """
        async with self._lock:
            if username in self._username_index:
                raise AuthenticationError(f"Username '{username}' already exists")
            
            if email in self._email_index:
                raise AuthenticationError(f"Email '{email}' already registered")
            
            password_hash, salt = PBKDF2Hasher.hash_password(password)
            
            user = User(
                id=generate_id(),
                username=username,
                password_hash=password_hash,
                salt=salt,
                email=email,
                roles=roles or ["user"],
                created_at=current_timestamp(),
                is_active=True
            )
            
            self._users[user.id] = user
            self._username_index[username] = user.id
            self._email_index[email] = user.id
            
            # Persist if storage available
            if self.storage:
                await self.storage.save("users", user.id, user.to_dict())
            
            return user
    
    async def authenticate(
        self, 
        username: str, 
        password: str
    ) -> Tuple[User, AuthToken]:
        """
        Authenticate a user and return tokens.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (User, AuthToken)
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        async with self._lock:
            user_id = self._username_index.get(username)
            if not user_id:
                raise AuthenticationError("Invalid username or password")
            
            user = self._users.get(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("Invalid username or password")
            
            if not PBKDF2Hasher.verify_password(
                password, user.password_hash, user.salt
            ):
                raise AuthenticationError("Invalid username or password")
            
            # Update last login
            user.last_login = current_timestamp()
            
            # Generate tokens
            token = await self.token_manager.generate_token_pair(
                user.id, 
                scopes=user.roles
            )
            
            return user, token
    
    async def validate_and_get_user(self, token: str) -> Optional[User]:
        """
        Validate a token and return the associated user.
        
        Args:
            token: Access token
            
        Returns:
            User if valid, None otherwise
        """
        auth_token = await self.token_manager.validate_token(token)
        if not auth_token:
            return None
        
        return self._users.get(auth_token.user_id)
    
    async def refresh_authentication(self, refresh_token: str) -> Optional[AuthToken]:
        """
        Refresh authentication using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New AuthToken if successful, None otherwise
        """
        return await self.token_manager.refresh_token(refresh_token)
    
    async def logout(self, token: str) -> bool:
        """
        Logout by revoking the token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if successful
        """
        return await self.token_manager.revoke_token(token)
    
    async def logout_all(self, user_id: str) -> int:
        """
        Logout from all sessions.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions logged out
        """
        return await self.token_manager.revoke_all_user_tokens(user_id)
    
    async def change_password(
        self, 
        user_id: str, 
        old_password: str, 
        new_password: str
    ) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful
            
        Raises:
            AuthenticationError: If old password is incorrect
        """
        async with self._lock:
            user = self._users.get(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            if not PBKDF2Hasher.verify_password(
                old_password, user.password_hash, user.salt
            ):
                raise AuthenticationError("Current password is incorrect")
            
            # Hash new password
            new_hash, new_salt = PBKDF2Hasher.hash_password(new_password)
            user.password_hash = new_hash
            user.salt = new_salt
            
            # Revoke all existing tokens
            await self.token_manager.revoke_all_user_tokens(user_id)
            
            return True
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_id = self._username_index.get(username)
        return self._users.get(user_id) if user_id else None
    
    async def update_user(self, user_id: str, updates: Dict) -> Optional[User]:
        """Update user fields."""
        async with self._lock:
            user = self._users.get(user_id)
            if not user:
                return None
            
            allowed_fields = {'email', 'roles', 'is_active', 'metadata'}
            for key, value in updates.items():
                if key in allowed_fields:
                    setattr(user, key, value)
            
            return user
    
    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        async with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            user.is_active = False
            await self.token_manager.revoke_all_user_tokens(user_id)
            return True
    
    async def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission/role."""
        user = self._users.get(user_id)
        if not user:
            return False
        return permission in user.roles or "admin" in user.roles


def require_auth(scopes: List[str] = None):
    """
    Decorator for requiring authentication on async functions.
    
    Args:
        scopes: Required permission scopes
    """
    def decorator(func):
        async def wrapper(self, *args, token: str = None, **kwargs):
            if not token:
                raise AuthorizationError("Authentication required")
            
            auth_token = await self.auth_service.token_manager.validate_token(token)
            if not auth_token:
                raise AuthorizationError("Invalid or expired token")
            
            if scopes:
                if not any(scope in auth_token.scopes for scope in scopes):
                    raise AuthorizationError("Insufficient permissions")
            
            return await func(self, *args, token=token, **kwargs)
        return wrapper
    return decorator