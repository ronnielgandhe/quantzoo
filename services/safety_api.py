"""
Safety API with kill switch for emergency trading shutdown.

⚠️ ⚠️ ⚠️ CRITICAL SAFETY SYSTEM ⚠️ ⚠️ ⚠️

This API provides:
1. Global kill switch to stop all trading immediately
2. Health monitoring endpoints
3. Emergency position closure
4. Audit logging of all safety actions

DO NOT DEPLOY WITHOUT:
- Proper authentication and authorization
- Encrypted communication (HTTPS)
- Rate limiting
- Comprehensive logging
- Backup communication channels
- Runbook for emergency procedures

This is a LAST RESORT safety mechanism. It does NOT replace:
- Proper risk management
- Position sizing
- Stop losses
- Regular monitoring
- Compliance oversight
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safety_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantZoo Safety API",
    description="Emergency kill switch and safety monitoring",
    version="1.0.0"
)

security = HTTPBearer()

# Import broker interface for kill switch
try:
    from connectors.brokers import BrokerInterface
except ImportError:
    logger.error("Failed to import BrokerInterface. Kill switch will not work.")
    BrokerInterface = None


# ============================================================================
# Authentication
# ============================================================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify API token.
    
    ⚠️ PRODUCTION: Replace with proper JWT verification, OAuth, or similar.
    This is a STUB for demonstration.
    """
    expected_token = os.getenv('SAFETY_API_TOKEN', 'development_token_change_me')
    
    if credentials.credentials != expected_token:
        logger.warning(f"Unauthorized access attempt with token: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    return credentials.credentials


# ============================================================================
# Models
# ============================================================================

class KillSwitchRequest(BaseModel):
    """Request to activate kill switch."""
    reason: str
    close_positions: bool = True
    operator: str


class KillSwitchResponse(BaseModel):
    """Response from kill switch activation."""
    success: bool
    timestamp: datetime
    stop_all_trading: bool
    message: str
    positions_closed: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    stop_all_trading: bool
    environment: str
    version: str


class SafetyStatus(BaseModel):
    """Current safety status."""
    stop_all_trading: bool
    last_updated: Optional[datetime]
    last_operator: Optional[str]
    last_reason: Optional[str]


# ============================================================================
# State Management
# ============================================================================

class SafetyState:
    """Global safety state."""
    def __init__(self):
        self.stop_all_trading = False
        self.last_updated: Optional[datetime] = None
        self.last_operator: Optional[str] = None
        self.last_reason: Optional[str] = None
        self.audit_log: List[Dict[str, Any]] = []
    
    def activate_kill_switch(self, reason: str, operator: str) -> None:
        """Activate kill switch."""
        self.stop_all_trading = True
        self.last_updated = datetime.now()
        self.last_operator = operator
        self.last_reason = reason
        
        # Update global flag in broker interface
        if BrokerInterface:
            BrokerInterface.STOP_ALL_TRADING = True
        
        # Audit log
        self.audit_log.append({
            'action': 'KILL_SWITCH_ACTIVATED',
            'timestamp': self.last_updated.isoformat(),
            'operator': operator,
            'reason': reason
        })
        
        logger.critical(f"🛑 KILL SWITCH ACTIVATED by {operator}: {reason}")
    
    def deactivate_kill_switch(self, operator: str) -> None:
        """Deactivate kill switch."""
        self.stop_all_trading = False
        self.last_updated = datetime.now()
        self.last_operator = operator
        
        # Update global flag
        if BrokerInterface:
            BrokerInterface.STOP_ALL_TRADING = False
        
        # Audit log
        self.audit_log.append({
            'action': 'KILL_SWITCH_DEACTIVATED',
            'timestamp': self.last_updated.isoformat(),
            'operator': operator
        })
        
        logger.warning(f"✅ Kill switch deactivated by {operator}")


# Global state
safety_state = SafetyState()


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - no authentication required."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        stop_all_trading=safety_state.stop_all_trading,
        environment=os.getenv('QUANTZOO_ENV', 'development'),
        version="1.0.0"
    )


@app.get("/status", response_model=SafetyStatus)
async def get_status(token: str = Depends(verify_token)):
    """Get current safety status."""
    return SafetyStatus(
        stop_all_trading=safety_state.stop_all_trading,
        last_updated=safety_state.last_updated,
        last_operator=safety_state.last_operator,
        last_reason=safety_state.last_reason
    )


@app.post("/kill-switch/activate", response_model=KillSwitchResponse)
async def activate_kill_switch(
    request: KillSwitchRequest,
    token: str = Depends(verify_token)
):
    """
    🛑 ACTIVATE KILL SWITCH 🛑
    
    This will:
    1. Set global STOP_ALL_TRADING flag to True
    2. Prevent all new orders from being placed
    3. Optionally close all open positions
    4. Log the action with operator and reason
    
    This is an EMERGENCY function. Use only when:
    - System malfunction detected
    - Excessive losses occurring
    - Regulatory requirement
    - Market conditions require immediate halt
    
    Requires authentication.
    """
    logger.critical(f"Kill switch activation requested by {request.operator}: {request.reason}")
    
    # Activate kill switch
    safety_state.activate_kill_switch(request.reason, request.operator)
    
    positions_closed = []
    
    # TODO: Close positions if requested
    # This requires access to active broker connections
    # For now, just set the flag
    if request.close_positions:
        logger.warning("Position closure requested but not yet implemented in this version")
        logger.warning("Positions must be closed manually via broker interface")
    
    return KillSwitchResponse(
        success=True,
        timestamp=datetime.now(),
        stop_all_trading=True,
        message=f"Kill switch activated by {request.operator}",
        positions_closed=positions_closed if request.close_positions else None
    )


@app.post("/kill-switch/deactivate")
async def deactivate_kill_switch(
    operator: str,
    token: str = Depends(verify_token)
):
    """
    Deactivate kill switch.
    
    ⚠️ Only deactivate after:
    1. Root cause identified and resolved
    2. System health verified
    3. Proper authorization obtained
    4. Monitoring confirmed operational
    
    Requires authentication.
    """
    logger.warning(f"Kill switch deactivation requested by {operator}")
    
    safety_state.deactivate_kill_switch(operator)
    
    return {
        "success": True,
        "timestamp": datetime.now(),
        "stop_all_trading": False,
        "message": f"Kill switch deactivated by {operator}"
    }


@app.get("/audit-log")
async def get_audit_log(
    limit: int = 100,
    token: str = Depends(verify_token)
):
    """Get audit log of safety actions."""
    return {
        "log": safety_state.audit_log[-limit:],
        "total_entries": len(safety_state.audit_log)
    }


@app.post("/test/safety-check")
async def test_safety_check(token: str = Depends(verify_token)):
    """
    Test endpoint to verify safety checks are working.
    
    Returns whether a hypothetical order would be blocked.
    """
    if BrokerInterface:
        # Create a dummy broker instance to test
        from connectors.brokers import PaperBroker
        test_broker = PaperBroker({'dry_run': True})
        would_pass = test_broker._safety_check()
    else:
        would_pass = not safety_state.stop_all_trading
    
    return {
        "safety_check_result": "PASS" if would_pass else "FAIL",
        "stop_all_trading": safety_state.stop_all_trading,
        "would_allow_orders": would_pass,
        "timestamp": datetime.now()
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup."""
    logger.info("="*80)
    logger.info("🚨 QUANTZOO SAFETY API STARTED 🚨")
    logger.info(f"Environment: {os.getenv('QUANTZOO_ENV', 'development')}")
    logger.info(f"Kill switch status: {'ACTIVE' if safety_state.stop_all_trading else 'INACTIVE'}")
    logger.info("="*80)
    
    # Check for authentication token
    token = os.getenv('SAFETY_API_TOKEN')
    if not token or token == 'development_token_change_me':
        logger.warning("⚠️  WARNING: Using default/development authentication token!")
        logger.warning("⚠️  Set SAFETY_API_TOKEN environment variable for production!")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("Safety API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    # Check environment
    env = os.getenv('QUANTZOO_ENV', 'development')
    if env == 'production':
        logger.warning("⚠️  Running in PRODUCTION mode")
        logger.warning("⚠️  Ensure proper authentication and HTTPS are configured!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,  # Different port from main API
        log_level="info"
    )
