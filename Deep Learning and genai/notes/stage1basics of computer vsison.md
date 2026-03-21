# Stage 1: Basics of Computer Vision

**Overview:** Stage 1 focuses on the fundamental concepts of how computers interpret visual data, the specific tasks they are trained to perform, and why this field is uniquely challenging compared to human vision.

## 1. The Goal of Computer Vision

The primary objective of computer vision is to make computers understand images and video. While a computer "sees" an image as a grid of numbers (pixels), a vision system aims to extract high-level "knowledge" from that data.

### Relationship with Other Fields

| Field | Transformation |
|-------|---|
| **Image Processing** | Image → New Image (e.g., sharpening a photo) |
| **Computer Graphics** | Knowledge/Models → Image |
| **Computer Vision** | Image → Knowledge (e.g., identifying a "cat" or "deer") |

**Example:** A system asking, "How far is that building?" or "Where are the cars?" to navigate a physical space.

**Formula:** Image → CV System → Knowledge


---

## 2. Core Visual Recognition Tasks

Recognition can be broken down into several distinct problems based on what the system needs to identify.

### Types of Recognition Tasks

1. **Image Classification:** Determining if a specific object is present in the image
   - Example: "Is there a car in this picture?"

2. **Object Detection:** Identifying what objects are present and where they are located, typically using a bounding box
   - Example: Draw boxes around all cars in a street scene

3. **Image Segmentation:** Assigning a label to every single pixel to determine exactly which pixels belong to which object
   - Example: Color only the car's paint and wheels

4. **Activity Recognition:** Moving beyond static objects to understand actions
   - Example: "What is this person doing?"

**Technical Formula (Classification):** $f(x) = y$, where $x$ is the image input and $y$ is a discrete class label

**Intuitive Example:** In a street scene:
- **Classification** identifies "Car"
- **Detection** draws a box around the car
- **Segmentation** masks only the car's paint and wheels


---

## 3. Why Recognition is Difficult

Unlike humans, computers struggle with visual data because the same object can look completely different depending on the environment. The following challenges are unique to computer vision:

### Key Challenges

- **Viewpoint Variation:** A chair looks different from the front than it does from above

- **Illumination:** Lighting changes the numerical values of pixels drastically, even if the object hasn't moved

- **Occlusion:** Only part of the object might be visible (e.g., a person standing behind a tree)

- **Deformation:** Many objects are not rigid (e.g., a running horse or a person sitting)

- **Intra-class Variation:** There are millions of different designs for a single category like "chair," yet a computer must recognize them all as the same thing

**Real-World Example:** A computer might fail to recognize a cup if it is:
- Viewed from directly above (**viewpoint variation**)
- Half-hidden by a napkin (**occlusion**)
- In shadow from harsh lighting (**illumination**)


---

## 4. From Geometric Era to Deep Learning

The history of recognition has evolved from manual, math-heavy models to data-driven learning. Understanding this progression helps explain why deep learning is so powerful today.

### Historical Evolution

| Era | Period | Approach |
|-----|--------|----------|
| **Geometric Era** | 1960s-1990s | Recognition as alignment problem using "geons" (geometric primitives like cubes and cylinders) to build object models |
| **Feature Engineering** | Pre-Deep Learning | Researchers manually defined "features" (discriminative parts of an image) and used them to train shallow classifiers |
| **Deep Learning** | Current | Network learns automatically to extract the best features directly from raw data (end-to-end learning) |

### Key Insight: Feature Learning

Instead of a human telling a computer to "look for circles to find wheels," a Deep Learning model discovers that circles are useful for identifying cars on its own.

### Traditional vs. Deep Learning Pipeline

**Traditional Approach:**
```
Input → Manual Feature Extraction → Shallow Classifier → Output
```

**Deep Learning Approach:**
```
Input → Deep Neural Network (Feature Learning + Classifier) → Output
```

The deep learning approach is superior because:
- ✅ No manual feature engineering required
- ✅ Features are learned from data
- ✅ End-to-end optimization possible
- ✅ Automatic discovery of useful representations