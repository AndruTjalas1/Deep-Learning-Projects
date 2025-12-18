# Contributing to Deep Learning Projects

Thank you for your interest in contributing to this portfolio project! While this is primarily a portfolio showcase, we welcome improvements, bug fixes, and suggestions.

## 🤝 How to Contribute

### Reporting Issues

1. Check the existing issues to avoid duplicates
2. Provide a clear description of the issue
3. Include:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU info if relevant)
   - Error messages or logs

### Suggesting Improvements

- Open an issue with the `enhancement` tag
- Describe what could be improved and why
- Include examples if applicable

### Submitting Pull Requests

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, descriptive commits
4. **Test thoroughly**:
   - Ensure existing functionality isn't broken
   - Add tests for new features if applicable
   - Verify on both CPU and GPU if relevant
5. **Push to your fork** and **create a Pull Request**

## 📋 Code Style Guidelines

### Python
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable names
- Add docstrings to functions and classes
- Use type hints where appropriate
- Maximum line length: 100 characters

Example:
```python
def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int = 10
) -> Dict[str, List[float]]:
    """
    Train a model for specified epochs.
    
    Args:
        model: The neural network model to train
        dataloader: DataLoader with training batches
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
    """
```

### JavaScript/React
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use functional components
- Use meaningful component and variable names
- Add comments for complex logic

### Markdown
- Use clear headings hierarchy
- Include code examples for setup instructions
- Keep lines readable (wrap at ~80 characters for text)

## 🎯 Development Setup

Each project has its own setup guide:

- **Handwriting Recognition**: See [Deep Neural Network/SETUP_GUIDE.md](./Deep%20Neural%20Network/SETUP_GUIDE.md)
- **DCGAN**: See [GAN/GETTING_STARTED.md](./GAN/GETTING_STARTED.md)
- **RNN**: See [RNN/backend/README.md](./RNN/backend/README.md)

## 🧪 Testing

Before submitting a PR:

1. **Python Projects**: Run linting and basic tests
   ```bash
   # Check for syntax errors
   python -m py_compile your_file.py
   
   # Format code
   pip install black
   black .
   ```

2. **React Projects**: Verify no console errors
   ```bash
   npm run dev
   # Check browser console for errors
   ```

## 📝 Commit Messages

Use clear, descriptive commit messages:

```
✨ Add feature: Brief description

Longer explanation of what was added and why, if needed.
```

Types:
- `✨ Add` - New features
- `🐛 Fix` - Bug fixes
- `📝 Docs` - Documentation
- `♻️ Refactor` - Code restructuring
- `🚀 Improve` - Performance improvements
- `🧹 Cleanup` - Remove dead code, clean up

## 🔍 Review Process

- PRs will be reviewed for:
  - Code quality and style
  - Functionality and correctness
  - Documentation completeness
  - No breaking changes to existing features
  
## ⚖️ License

By contributing, you agree that your contributions will be licensed under the same MIT License as the project.

## 🎓 Learning Resources

If you're new to the projects:

- **PyTorch**: [Official Tutorials](https://pytorch.org/tutorials/)
- **FastAPI**: [Official Documentation](https://fastapi.tiangolo.com/)
- **React**: [Official Docs](https://react.dev/)
- **TensorFlow**: [Official Guides](https://www.tensorflow.org/guide)

## 💡 Ideas for Contribution

- Add unit tests for Python modules
- Improve error messages and validation
- Optimize model inference speed
- Add new animal support to DCGAN
- Enhance UI/UX of React frontends
- Improve documentation and examples
- Add performance benchmarks
- Improve mobile responsiveness

## ❓ Questions?

Open an issue with your question or check existing issues for similar questions.

---

**Thank you for contributing!** Your efforts help make these projects better for everyone. 🙏
