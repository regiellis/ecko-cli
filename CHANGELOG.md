# Changelog

## All notable changes to this project will be documented in this file


##  [1.5.0] - 2024-10-07
- **Model Update**: replaced Florence with a new model, MiaoshouAI/Florence-2-base-PromptGen-v2.0. This model is fine-tune of Florence2

##  [1.4.1] - 2024-10-07
- **Hotfix**: Update caption mode for JoyCaption, fix some typoes

### [1.4.0] - 2024-10-07

- **Caption Features**: Added JoyCaption Alpha to options for captioning. This is a new feature that allows for richer captions to be generated.
- **Caption UI**: Added a new gradio UI for editing captions. This allows for more control over the captioning process.

### [1.3.0] - 2024-09-30

- **Caption Features**: New caption options and the abailty to install flash-attn from the command line.

### [1.2.0] - 2024-09-23

- **Dateset Features**: Now allows for any popular dataset file to be exported for training. This includes `JSONL`, `HFJSON`, `CSV`, and `JSON`.

### [1.1.0] - 2024-09-22

- **Images Features**: Now generates two sets of images at 512, 1024. training is done on a 672x672 image.

### [1.0.0] - 2024-08-27

- **Initial Release**: Established the initial codebase for the project, laying the foundation for future development and enhancements.
