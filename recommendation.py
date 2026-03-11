solutions = {

"Early_Blight":
"Fungal infection. Remove infected leaves and apply fungicide.",

"Late_Blight":
"Serious disease. Apply copper fungicide and avoid wet leaves.",

"Leaf_Mold":
"Improve ventilation and reduce humidity.",

"Septoria":
"Remove infected leaves and rotate crops.",

"Healthy":
"Leaf appears healthy. Maintain good watering."

}

def get_solution(disease):

    if disease == "Unknown / Not Tomato Leaf":
        return "The system cannot detect a known tomato disease."

    return solutions.get(disease,"No recommendation available.")