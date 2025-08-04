import pytest
from expandor.core.configuration_manager import ConfigurationManager

class TestFailLoudConfiguration:
    """Test that configuration fails loud on missing keys"""
    
    def test_missing_key_fails_loud(self):
        """Missing keys should raise ValueError, not return defaults"""
        cm = ConfigurationManager()
        
        with pytest.raises(ValueError) as exc_info:
            cm.get_value('nonexistent.key.that.should.fail')
        
        assert "not found" in str(exc_info.value)
        assert "Solutions:" in str(exc_info.value)
    
    def test_scientific_notation_types(self):
        """Scientific notation should parse as float"""
        cm = ConfigurationManager()
        
        epsilon = cm.get_value('strategies.tiled_expansion.division_epsilon')
        assert isinstance(epsilon, float)
        assert epsilon == 1e-8
    
    def test_no_silent_defaults(self):
        """Ensure no .get() with defaults in critical paths"""
        import ast
        import inspect
        from expandor.core import configuration_manager
        
        source = inspect.getsource(configuration_manager)
        tree = ast.parse(source)
        
        # Find all .get() calls with defaults
        class GetChecker(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
            
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr == 'get' and 
                    len(node.args) > 1):
                    # Exclude os.environ.get() which is legitimately optional
                    code = ast.unparse(node)
                    if 'os.environ.get' not in code:
                        self.violations.append(code)
                self.generic_visit(node)
        
        checker = GetChecker()
        checker.visit(tree)
        
        # Should have no .get() with defaults in configuration manager (except env vars)
        assert len(checker.violations) == 0, f"Found .get() with defaults: {checker.violations}"