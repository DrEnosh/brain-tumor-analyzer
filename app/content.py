BODY_PART = "Brain"
IMAGING = "Magnetic resonance imaging (MRI), single 2D slice"

CLASS_INFO = {
    "glioma": {
        "origin": "Glial cells inside the brain tissue",
        "description": "Gliomas grow from glial cells, the cells that support and protect neurons, inside the brain "
                       "tissue itself. In adults they most often occur in the cerebral hemispheres and can spread "
                       "into the surrounding brain tissue.",
    },
    "meningioma": {
        "origin": "Meninges (membranes covering the brain)",
        "description": "Meningiomas grow from the meninges, the protective membranes that wrap the brain and spinal "
                       "cord. They usually sit on the surface of the brain, just under the skull, and press inward on "
                       "the brain rather than growing inside it. Most are slow-growing and benign.",
    },
    "pituitary": {
        "origin": "Pituitary gland at the base of the brain",
        "description": "Pituitary tumours (usually adenomas) grow in the pituitary gland, a pea-sized hormone gland at "
                       "the base of the brain. It sits in a bony pocket called the sella turcica, just below where the "
                       "optic nerves cross. Most are benign but can affect hormone levels or vision.",
    },
    "notumor": {
        "origin": "No tumour pattern detected",
        "description": "The model did not find the patterns of a glioma, meningioma or pituitary tumour in this slice. "
                       "It only recognises these three tumour types, so other conditions are not ruled out.",
    },
}

DISCLAIMER = ("Research prototype for educational use only. It is not a medical device, and its output must not be "
              "used for diagnosis or treatment decisions. Every result must be reviewed by a qualified radiologist.")
