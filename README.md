# AXLERO INTERNSHIP — GROUP 1 PROJECT CONTEXT

## 1. Internship and Team

- Organization: Axlero Innovative Solutions
- Internship role: Data Science Intern
- Internship started: 6 October 2026
- Team size: 5 members
- Team leader and repository owner: M Harshavardhana

### Team members
1. M Harshavardhana — Team Leader
2. Sri Chidvilas Govindarajula
3. Dadiga Sandhya
4. Devraj Giri
5. Polepalli Sai Sushmanth

## 2. GitHub Repository Setup

Repository: https://github.com/harsha282004/axlero-internship

The repository is owned by Harshavardhana and contains three project folders:

1. `project-1-healthcare`
2. `project-2- Manufacturing Predictive Maintenance...`
3. `project-3-Agriculture Crop Yield Forecasting`

The repository is currently public.

All four teammates have been added as GitHub collaborators. Branch names are flexible; each member may choose their own branch name.

### Collaboration workflow
1. Harsha assigns work to each member.
2. Members work on their individual branches.
3. Members commit and push their changes to their respective branches.
4. Members open Pull Requests targeting `main`.
5. Harsha reviews the changes and requests corrections if necessary.
6. Harsha merges approved changes into `main`.

Pushing to an individual branch does not automatically update `main`.

### Repository settings completed
- Repository description added.
- A branch ruleset named `Protect main` has been created and is Active.
- The ruleset targets the default branch.
- Pull Requests are required before merging.
- One approving review is required.
- Restrict deletions is enabled.
- The bypass list is empty.

The next step is to verify the saved ruleset if needed and continue configuring the repository documentation.

## 3. Project 1 — Healthcare Image Classification

Project title: Healthcare & Medical Diagnostic Lab Image Classification Using Computer Vision.

### Main objectives
- Select a suitable public medical imaging dataset.
- Perform exploratory data analysis and data-quality checks.
- Build preprocessing and image-loading pipelines.
- Train a pretrained CNN baseline, initially considering ResNet50.
- Evaluate using recall for the positive/critical class, precision, F1-score and confusion matrix.
- Track experiments using MLflow or Weights & Biases.
- Implement Grad-CAM explainability.
- Build an inference API using FastAPI or Flask.
- Document the code, experiments, results and limitations.

The Axlero project requirements specify a target of greater than 90% recall for the positive/critical class. This is a target, not a guaranteed result.

The project is educational and must not be presented as a clinically validated diagnostic system.

## 4. Two-Week Work Allocation

The team is following an accelerated 14-day internal plan.

### Harshavardhana — Team Leader and Lead Developer
- Own project architecture and component interfaces.
- Coordinate the dataset, preprocessing, training and API pipeline.
- Run and integrate the baseline.
- Review Pull Requests and resolve integration problems.
- Select the final model using validation data.
- Verify metrics, tests, documentation and final delivery.

Branch suggestion: `p1-harsha`

### Sri Chidvilas Govindarajula — Dataset and EDA
- Identify a suitable public dataset.
- Document source, licence, labels and limitations.
- Produce class counts and exploratory plots.
- Check corrupted files, duplicates and data leakage risks.
- Document dataset splits and provenance.

Suggested branch: `p1-sri-chidvilas`

### Dadiga Sandhya — Preprocessing and Augmentation
- Implement PyTorch Dataset/DataLoader.
- Implement resizing, normalization and transformations.
- Add training-only augmentation.
- Validate tensor shapes, labels and error handling.
- Ensure inference preprocessing matches evaluation preprocessing.

Suggested branch: `p1-sandhya`

### Devraj Giri — Model Training and Evaluation
- Implement a pretrained CNN baseline.
- Run training and save checkpoints/configuration.
- Calculate recall, precision, F1-score and confusion matrix.
- Run controlled fine-tuning experiments.
- Select checkpoints using validation data only.
- Document actual model performance.

Suggested branch: `p1-devraj`

### Polepalli Sai Sushmanth — Tracking, Grad-CAM and API
- Set up experiment tracking.
- Create evaluation report templates.
- Implement Grad-CAM visualizations.
- Build the inference API and input validation.
- Write API tests and documentation.
- Ensure reported metrics are traceable to experiment runs.

Suggested branch: `p1-sushmanth`

These branch names are suggestions, not mandatory names.

## 5. Two-Week Milestones

### Week 1
- Confirm dataset, labels, class mapping and interfaces.
- Implement dataset loader and preprocessing.
- Run the first end-to-end baseline.
- Validate splits and metrics.
- Produce baseline checkpoint, training curves and initial Grad-CAM/API components.
- Review and integrate working code.

### Week 2
- Run controlled model-improvement experiments.
- Select the best candidate using validation data.
- Integrate Grad-CAM and the inference API.
- Run tests and reproducibility checks.
- Finalize the README, report, results and presentation.
- Review Pull Requests and merge approved work.

Important: The Axlero handbook specifies a four-week implementation cycle. The two-week plan is an accelerated internal schedule, so the shortened timeline should be confirmed with the internship coordinator.

## 6. Current Immediate Next Step

We are configuring the repository step by step before the team begins development.

The repository description has been saved successfully. The next task is to update the root `README.md`.

Instructions:
1. Open the repository Code page.
2. Find the large README section below the three project folders.
3. Click the pencil icon at the top-right of that README section.
4. Open the editor and prepare to replace the minimal existing README with a complete overview of the repository.

The root README should explain:
- The internship and Group 1.
- All three project folders.
- The five team members.
- The branch and Pull Request workflow.
- The rule that Harsha reviews and merges approved work into `main`.

Make this change on `main` because the README describes the entire repository.

## 7. How to Continue

Continue guiding Harshavardhana one step at a time. Do not jump ahead or provide many unrelated settings at once. After he opens the README editor, provide the complete Markdown content to paste and then guide him through saving it.

Keep all Project 1 work inside `project-1-healthcare/`, and do not assume that folder-specific branches exist: Git branches apply to the repository, not individual folders.

The immediate goal is to finish repository preparation, share the project work-plan PDF with the team, confirm the dataset and interfaces, and begin the two-week implementation plan.
