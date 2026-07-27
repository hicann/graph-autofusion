# Contributing Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, see [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and understand the contribution process of the source code repository. This repository details the prerequisites for participating in CANN open source project contributions, including but not limited to:
1. How to submit a PR
2. Gitcode workflow
3. Pipeline trigger commands
4. Code review
5. Other notes
For details, see [cann-community](https://gitcode.com/cann/community).

In addition, developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, carefully fill in the business background, purpose, solution, and other information according to the PR template.
2. If your modification is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modified code flow, discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also discuss the solution by submitting an Issue.
3. When submitting a PR, ensure that your code conforms to the project code specifications. For details, see Google's [Open Source Code Style Guide](https://google.github.io/styleguide/), including but not limited to:
   - Code formatting
   - Comment specifications
   - Variable naming specifications
   - Function naming specifications
   - Class naming specifications
   - Interface naming specifications
   - Configuration parameter naming specifications
   - Code flow specifications
4. The pipeline runs a code check and rejects code that has not been formatted by clang-format through pre-commit. We recommend installing pre-commit and running `pre-commit install` in the repository root so that each local commit is automatically checked and formatted.
5. When submitting a PR, if there are multiple invalid commits, it is recommended that you perform a rebase operation before submitting the PR to merge multiple commits into one to maintain code simplicity and readability. For details, see [git rebase](https://git-scm.com/docs/git-rebase). At the same time, the commit message must also conform to the project code specifications and clearly describe the intent and content of this change. The format is: <type>: <brief description>. For example:

|Type|Description|Example|
|--|--|--|
|feat|New feature|feat: Add user registration function|
|fix|Bug fix|fix: Fix login session expiration issue|
|docs|Documentation update|docs: Update API usage instructions|
|style|Code format adjustment (does not affect logic)|style: Adjust code indentation|
|refactor|Refactoring (non-feature addition/fix)|refactor: Optimize user service class structure|
|perf|Performance optimization|perf: Reduce database query count|
|test|Test related|test: Add login function unit test|
|chore|Build/toolchain change|chore: Update webpack configuration|
|ci|CI configuration related|ci: Add automated test process|

Developer contribution scenarios mainly include:

- Bug Fix

  If you discover certain bugs in this project and want to fix them, you are welcome to create an Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Bug-Report|Defect Feedback` type Issue to describe the bug, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for handling.


- Contribute New Features

  If you discover certain missing features in this project and want to add them, you are welcome to create an Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Requirement|Feature Suggestion` type Issue to describe the new feature and provide your design solution, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking implementation.


- Document Correction

  If you discover certain document description errors in this project, you are welcome to create an Issue for feedback and correction.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Documentation|Document Feedback` type Issue to point out the corresponding document problem, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself to correct the corresponding document description.

- Help Solve Others' Issues

  If you have appropriate solutions for problems encountered by others in the community, you are welcome to post comments in the Issue to exchange ideas, help others solve problems and pain points, and jointly optimize usability.

  If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself to track and assist in solving the problem.
