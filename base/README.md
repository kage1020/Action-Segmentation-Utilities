# Base class for ASU

This class provides utility functions for the ASU project.

| Function | Static | Description |
| --- | --- | --- |
| `init_seed` | ✅ | Initialize the seed for reproducibility. |
| `get_device` | ✅ | Get the cpu/cuda device for the model. |
| `get_logger` | ✅ | Get the logger for the model. |
| `get_time` | ✅ | Get the current time. |
| `get_elapsed_time` | ✅ | Get the elapsed time. |
| `log_time` | ✅ | Log the elapsed time with format. |
| `validate_config` | ✅ | [Not Implemented] Validate the configuration. |
| `load_model` | ✅ | Load the model from the checkpoint. |
| `load_best_model` | ✅ | Load the best model from the checkpoint. |
| `save_model` | ✅ | Save the model to the checkpoint. |
| `to_np` | ✅ | Convert the list/tensor to numpy. |
| `to_torch` | ✅ | Convert the list/numpy to tensor. |
| `to_class_name` | ✅ | Convert numpy/tensor of class index to class name. |
| `to_class_index` | ✅ | Convert class name to class index. |
| `get_image_paths` | ✅ | Get the image paths from the directory. |
| `load_image` | ✅ | Load the image from the path. |
| `load_images` | ✅ | Load the images from the directory. |
| `get_class_mapping` | ✅ | Get the class mapping from the file. |
| `set_class_mapping` |  | Set the class mapping to variable. |
| `update_class_mapping` |  | Update the class mapping to variable. |
| `mask_mapping_with_backgrounds` |  | Mask the class mapping with backgrounds. |
| `get_actions` | ✅ | Get the actions from the file. |
| `set_actions` |  | Set the actions to variable. |
| `mask_actions_with_backgrounds` |  | Mask the actions with backgrounds. |
| `get_action_matching` | ✅ | Get the action matching from the file. |
| `set_action_matching` |  | Set the action matching to variable. |
| `to_segments` | ✅ | Convert the frame-wise labels to segments. |
| `mask_label_with_backgrounds` | ✅ | Mask the label with backgrounds. |
