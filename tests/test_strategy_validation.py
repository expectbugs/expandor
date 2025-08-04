import pytest
from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.config import ExpandorConfig

class TestStrategyValidation:
    """Test strategy input validation"""
    
    def test_none_seed_handled(self):
        """None seed should not cause TypeError"""
        config = ExpandorConfig(
            source_image="test.jpg",
            target_resolution=(1024, 1024),
            seed=None  # This should not crash
        )
        
        # Create a minimal strategy implementation
        class TestStrategy(BaseExpansionStrategy):
            def execute(self, config, context=None):
                return {"success": True}
        
        strategy = TestStrategy(config=None, metrics=None, logger=None)
        # Should not raise
        strategy.validate_inputs(config)
    
    def test_invalid_resolution_rejected(self):
        """Invalid resolutions should be rejected"""
        config = ExpandorConfig(
            source_image="test.jpg",
            target_resolution=(0, 0),  # Invalid
            seed=42
        )
        
        class TestStrategy(BaseExpansionStrategy):
            def execute(self, config, context=None):
                return {"success": True}
        
        strategy = TestStrategy(config=None, metrics=None, logger=None)
        
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_inputs(config)
        
        assert "Invalid target resolution" in str(exc_info.value)