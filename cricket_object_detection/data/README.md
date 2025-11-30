# Dataset README

## Image Sources

This dataset is intended to be populated with cricket images from the following sources:

### Recommended Sources (with proper attribution/licensing):
1. **Sports Photography Websites**
   - Unsplash (https://unsplash.com) - Free to use
   - Pexels (https://pexels.com) - Free to use
   - Pixabay (https://pixabay.com) - Free to use

2. **Cricket Match Screenshots**
   - Personal recordings of cricket matches
   - Publicly available highlight clips (with permission)

3. **User-Captured Photographs**
   - Personal cricket game photos
   - Practice session photos

4. **Educational Resources**
   - Cricket coaching materials
   - Sports education websites

### Image Categories
- **bat/**: Images prominently featuring cricket bats (various angles, held by players, on ground)
- **ball/**: Images showing cricket balls (on ground, in air, in hand, red/white balls)
- **stumps/**: Images showing cricket stumps/wickets (clear view, during play)
- **no_object/**: Background images (pitch, grass, crowd, ground without equipment)

### Collection Guidelines
1. Collect at least 75 images per category (300 total minimum)
2. Ensure variety in angles, lighting, and contexts
3. Include both clear and partially occluded objects
4. Mix of indoor and outdoor settings
5. Various cricket formats (Test, ODI, T20)

### Image Requirements
- Minimum resolution: 800x600 pixels before preprocessing
- Aspect ratio: Will be cropped to 4:3 if needed
- Format: JPG, PNG, or other common formats
- No watermarks or logos covering key areas

### Annotation Instructions
After placing images in the appropriate folders:
1. Run the preprocessing script to ensure 800x600 resolution
2. Create annotations using the annotation tool
3. Mark grid cells containing objects with appropriate labels:
   - 0 = no object
   - 1 = ball
   - 2 = bat  
   - 3 = stump

### Legal Notice
All images used must comply with copyright laws. Ensure you have the right to use any images included in this dataset. For educational purposes only.
