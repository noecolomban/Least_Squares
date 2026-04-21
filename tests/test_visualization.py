import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import Visualization
from scheduled import WSDSchedule, ConstantSchedule
from src.new_schedules.polynomial import PolynomialSchedule

@pytest.fixture
def visualization_setup():
    """Fixture to instantiate a standard Visualization."""
    schedules = [
        WSDSchedule(steps=10, base_lr=0.1, cooldown_len=0.5),
        ConstantSchedule(steps=10, base_lr=0.05),
        PolynomialSchedule(steps=10, base_lr=0.01, exponent=0.5)
    ]
    schedules_names = ["wsd", "constant", "polynomial"]
    return Visualization(schedules, schedules_names)

def test_visualization_initialization(visualization_setup):
    """Test correct initialization of Visualization."""
    vis = visualization_setup
    
    assert len(vis.schedules) == 3
    assert "wsd" in vis.schedules
    assert "constant" in vis.schedules
    assert "polynomial" in vis.schedules
    assert len(vis.colors) == 3

def test_visualization_colors_unique(visualization_setup):
    """Test that colors are unique for each schedule."""
    vis = visualization_setup
    
    colors = list(vis.colors.values())
    assert len(set(colors)) == len(colors)  # All colors are unique

def test_visualization_colors_format(visualization_setup):
    """Test that colors are in valid hex format."""
    vis = visualization_setup
    
    for color in vis.colors.values():
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7  # Format hex standard: #RRGGBB

def test_visualization_default_names():
    """Test that default names are used if not provided."""
    schedules = [
        WSDSchedule(steps=5, base_lr=0.1, cooldown_len=0.5),
        ConstantSchedule(steps=5, base_lr=0.05)
    ]
    vis = Visualization(schedules)
    
    assert len(vis.schedules_names) == 2
    assert vis.schedules_names[0] == "wsd"
    assert vis.schedules_names[1] == "constant"

def test_make_filename(visualization_setup):
    """Test filename generation."""
    vis = visualization_setup
    
    filename = vis._make_filename("test_plot")
    assert "test_plot" in filename
    assert filename.endswith(".pdf")
    assert "images" in filename

# Note: Tests for plot_for_every_schedule and plot_comparison would require
# mocking matplotlib or managing figures, which is complex in a test environment.
# These methods are primarily wrappers around matplotlib.