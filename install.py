import launch

if not launch.is_installed("sklearn"):
    launch.run_pip("install scikit-learn", "scikit-learn")

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")

if not launch.is_installed("xformers"):
    launch.run_pip("install xformers==0.0.20", "xformers==0.0.20")

if not launch.is_installed("toml"):
    launch.run_pip("install toml==0.10.2", "toml==0.10.2")
