#!/bin/bash

# NeuPRE Quick Start Script

echo "======================================================================"
echo "NeuPRE: Neuro-Symbolic Protocol Reverse Engineering"
echo "======================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✓ Setup complete!"
echo ""
echo "======================================================================"
echo "Quick Start Options:"
echo "======================================================================"
echo ""
echo "1. Run example (recommended for first-time users):"
echo "   python example.py"
echo ""
echo "2. Run experiments:"
echo "   python main.py experiment 1  # State coverage efficiency"
echo "   python main.py experiment 2  # Field boundary accuracy"
echo "   python main.py experiment 3  # Complex constraint inference"
echo ""
echo "3. Analyze custom protocol:"
echo "   python main.py run -i path/to/messages -o output_dir"
echo ""
echo "4. View documentation:"
echo "   cat README.md"
echo "   cat USAGE.md"
echo ""
echo "======================================================================"
echo ""

# Optionally run example
read -p "Would you like to run the example now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running example..."
    python example.py
fi
