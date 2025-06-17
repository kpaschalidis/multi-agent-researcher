import sys

sys.path.insert(0, "..")


def test_core_imports():
    """Test that core components can be imported"""
    print("🧪 Testing core imports...")

    try:
        from framework import TaskComplexity, AgentLogger, LogLevel

        print("✅ Core types and logging imported successfully")

        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        print("✅ TaskComplexity enum works correctly")

        logger = AgentLogger(verbose=False)
        assert len(logger.logs) == 0
        print("✅ AgentLogger created successfully")

        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False


def test_framework_availability():
    """Test framework availability detection"""
    print("\n🧪 Testing framework availability...")

    try:
        from framework import check_framework_availability

        is_available, error = check_framework_availability()

        print(f"Framework available: {is_available}")
        if not is_available:
            print(f"Expected error (missing deps): {error}")

        return True
    except Exception as e:
        print(f"❌ Availability check failed: {e}")
        return False


def test_conditional_imports():
    """Test that conditional imports work correctly"""
    print("\n🧪 Testing conditional imports...")

    try:
        # These should fail gracefully
        from framework import create_general_research_workflow

        # This should raise an ImportError due to missing dependencies
        try:
            create_general_research_workflow(None, None)
            print("❌ Expected ImportError was not raised")
            return False
        except ImportError as e:
            print(f"✅ Expected ImportError raised: {str(e)[:60]}...")

        return True
    except Exception as e:
        print(f"❌ Conditional imports test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🎯 Testing Reorganized Multi-Agent Research Framework")
    print("=" * 60)

    tests = [
        test_core_imports,
        test_framework_availability,
        test_module_structure,
        test_conditional_imports,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The reorganized framework is working correctly.")
        print("\n📋 Framework Benefits:")
        print("- ✅ Modular structure with clear separation of concerns")
        print("- ✅ Core components work without external dependencies")
        print("- ✅ Graceful handling of missing dependencies")
        print("- ✅ Clean import structure")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
