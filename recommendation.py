solutions = {

"Early_Blight":
"Remove infected leaves and apply fungicide.",

"Late_Blight":
"Use copper fungicide and avoid high humidity.",

"Leaf_Mold":
"Improve air circulation around plants.",

"Septoria":
"Remove infected leaves and rotate crops.",

"Healthy":
"Leaf is healthy. Maintain proper watering."
}

def get_solution(disease):
    return solutions.get(disease,"No suggestion available")