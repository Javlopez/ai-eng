# DIAGNOSTIC SCRIPT FOR OPENAI ISSUES
# ===================================

import sys
import subprocess
import os


def run_diagnostic():
    """Run complete diagnostic for OpenAI setup"""

    print("🔍 OPENAI DIAGNOSTIC SCRIPT")
    print("=" * 50)

    # 1. Check Python version
    print(f"🐍 Python version: {sys.version}")

    # 2. Check current working directory
    print(f"📁 Current directory: {os.getcwd()}")

    # 3. Check if we're in a virtual environment
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"🌐 Virtual environment: {venv}")
    else:
        print("🌐 Virtual environment: Not detected")

    # 4. List installed packages related to openai
    print("\n📦 Checking installed packages...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                                capture_output=True, text=True)
        packages = result.stdout

        openai_packages = [line for line in packages.split('\n') if 'openai' in line.lower()]
        if openai_packages:
            print("   OpenAI related packages found:")
            for pkg in openai_packages:
                print(f"   ✅ {pkg}")
        else:
            print("   ❌ No OpenAI packages found")

    except Exception as e:
        print(f"   ❌ Error checking packages: {e}")

    # 5. Try different import methods
    print("\n🔄 Testing imports...")

    # Test 1: Direct import
    try:
        import rag_openai
        print(f"   ✅ import openai - SUCCESS (version: {openai.__version__})")
    except ImportError as e:
        print(f"   ❌ import openai - FAILED: {e}")
    except AttributeError:
        print("   ⚠️  import openai - SUCCESS but no __version__ attribute")

    # Test 2: OpenAI client import
    try:
        from rag_openai import OpenAI
        print("   ✅ from openai import OpenAI - SUCCESS")
    except ImportError as e:
        print(f"   ❌ from openai import OpenAI - FAILED: {e}")

    # Test 3: Legacy import
    try:
        import rag_openai
        if hasattr(openai, 'ChatCompletion'):
            print("   ✅ Legacy OpenAI API detected")
        else:
            print("   ℹ️  New OpenAI API detected (no ChatCompletion)")
    except:
        pass

    # 6. Check API key
    print("\n🔑 Checking API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"   ✅ OPENAI_API_KEY found: {masked_key}")
    else:
        print("   ❌ OPENAI_API_KEY not found in environment")

    # 7. Installation recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 30)

    try:
        import rag_openai
        from rag_openai import OpenAI
        print("✅ OpenAI is properly installed")

        if not api_key:
            print("🔧 Set your API key:")
            print("   export OPENAI_API_KEY='your-key-here'")

    except ImportError:
        print("🔧 Install OpenAI with one of these commands:")
        print("   pip install openai")
        print("   pip install openai>=1.0.0")
        print("   python -m pip install openai")
        print("\n🔧 If in virtual environment, make sure it's activated:")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")


def install_openai():
    """Try to install OpenAI automatically"""
    print("\n🔧 Attempting to install OpenAI...")

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai>=1.0.0'])
        print("✅ OpenAI installation completed")

        # Test the installation
        try:
            from rag_openai import OpenAI
            print("✅ OpenAI import test successful")
            return True
        except ImportError as e:
            print(f"❌ OpenAI import still failing: {e}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


def test_openai_connection():
    """Test actual connection to OpenAI API"""
    print("\n🌐 Testing OpenAI API connection...")

    try:
        from rag_openai import OpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Cannot test connection: No API key")
            return False

        client = OpenAI(api_key=api_key)

        # Test with a simple embedding call
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="test connection"
        )

        print("✅ OpenAI API connection successful!")
        print(f"   Response type: {type(response)}")
        print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
        return True

    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Run diagnostic
    run_diagnostic()

    # Ask user what to do
    print("\n" + "=" * 50)
    print("ACTIONS:")
    print("1. Try to install OpenAI automatically")
    print("2. Test OpenAI API connection")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        success = install_openai()
        if success:
            print("\n🎉 OpenAI installed successfully!")
            print("Now you can run your RAG code.")
        else:
            print("\n❌ Installation failed. Try manual installation:")
            print("pip install openai>=1.0.0")

    elif choice == "2":
        test_openai_connection()

    elif choice == "3":
        print("Exiting diagnostic.")
    else:
        print("Invalid choice.")

    print("\n📋 SUMMARY:")
    print("- Run this diagnostic anytime you have OpenAI issues")
    print("- Make sure you're in the right virtual environment")
    print("- Set OPENAI_API_KEY before testing connections")