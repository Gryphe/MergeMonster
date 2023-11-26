def calculate_final_composition(layer_origins):
    final_composition = {}

    for layer_idx, merges in layer_origins.items():
        current_composition = {}

        for ratio, model_name in merges:
            # Update contributions of existing models
            for existing_model in current_composition:
                current_composition[existing_model] *= (1 - ratio)

            # Add/Update the new model's contribution
            if model_name in current_composition:
                current_composition[model_name] += ratio
            else:
                current_composition[model_name] = ratio

        # Normalize the ratios (optional)
        total_ratio = sum(current_composition.values())
        for model_name in current_composition:
            current_composition[model_name] /= total_ratio

        final_composition[layer_idx] = current_composition

    return final_composition

def aggregate_composition(final_layer_composition):
    aggregated_composition = {}

    for layer_composition in final_layer_composition.values():
        for model_name, ratio in layer_composition.items():
            aggregated_composition[model_name] = aggregated_composition.get(model_name, 0) + ratio

    # Normalize the aggregated ratios
    total_ratio = sum(aggregated_composition.values())
    for model_name in aggregated_composition:
        aggregated_composition[model_name] /= total_ratio

    # Sort the dictionary by values (ratios) in descending order
    aggregated_composition = {k: v for k, v in sorted(aggregated_composition.items(), key=lambda item: item[1], reverse=True)}

    return aggregated_composition