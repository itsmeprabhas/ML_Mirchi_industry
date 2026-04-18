**ML Mirchi Dataset Card**

**Dataset Overview**


**Property **             **Value**
  Task	        Image Classification (4 classes)
  Subjects	    Chilli peppers (various Indian varieties)
  Format	      JPEG/PNG images
  Size	        2000 synthetic (dev) / 6000+ real (production)

**Classes**

**Class 	 Count        (Synthetic)	Description**
Grade_A	 500	   Premium quality — deep red, uniform, no defects
Grade_B	 500	   Standard — minor variation, slight imperfections
Grade_C	 500	   Sub-standard — discoloration, wrinkles, spots
Reject	 500	   Rejected — rot, mold, cracks, pest damage

**Splits**

Train: 70% (1400 images)
Validation: 15% (300 images)
Test: 15% (300 images)

**Collection Method (Real Data)**

**Development Phase**

Synthetic images generated procedurally with controlled parameters per grade.

**Production Phase**

Captured at factory sorting station with controlled LED lighting (5000K)
Black background, fixed camera mount at 30cm
Color calibration card in first 10 shots of each session
Varieties: Guntur Sannam, Byadgi, Kashmiri, Jwala, Bird's Eye
Edge cases deliberately included (borderline grades, mixed defects)

**Augmentation Applied (Training Only)**

Rotation, flip, shift-scale, brightness/contrast, hue/saturation/value, blur, noise, coarse dropout.

**Ethical Considerations**

No personal data involved
Images are of agricultural products only
No human subjects in frames
