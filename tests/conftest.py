def pytest_addoption(parser):
    parser.addoption(
        "--target", 
        action="store", 
        default="./version/serial/src/docs",
        help="Path to the C++ binary to test"
    )