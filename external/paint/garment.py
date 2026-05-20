


def predict_points(input_vertices, callback):
    output_vertices =input_vertices[:len(input_vertices)//2]  
    callback(output_vertices)
