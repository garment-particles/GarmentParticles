import os, glob, yaml, tqdm
gcd_folder = "/orion/u/w4756677/garment/gcdv2/garmentcodedatav2/garments_5000_*/default_body"
gcd_folders = glob.glob(os.path.join(gcd_folder, "rand_*"))
import json
print(len(gcd_folders))

upper_name_dict = {
    "Shirt": "shirt",
    "FittedShirt": "fitted shirt",
}

belt_name_dict = {
    "StraightWB": "straight waistband",
    "FittedWB": "fitted waistband",
}

lower_name_dict = {
    "SkirtCircle": "circle skirt",
    "AsymmSkirtCircle": "asymmetric circle skirt",
    "GodetSkirt": "godet skirt",
    "Pants": "pants",
    "Skirt2": "2-panel skirt",
    "SkirtManyPanels": "flare skirt",
    "PencilSkirt": "pencil skirt",
    "SkirtLevels": "tiered skirt",
}

collar_comp_dict = {
    "Turtle": "turtle neck",
    "SimpleLapel": "lapel neck",
    "Hood2Panels": "with hood",
}
collar_type_dict = {
    "CircleNeckHalf": "circular collar",
    "CurvyNeckHalf": "curvy collar",
    "VNeckHalf": "v-neck collar",
    "SquareNeckHalf": "square collar",
    "TrapezoidNeckHalf": "trapezoid collar",
    "CircleArcNeckHalf": "circular collar",
    "Bezier2NeckHalf": "curved collar",
}
armhole_shape_dict = {
    "ArmholeSquare": "square armhole",
    "ArmholeAngle": "angled armhole",
    "ArmholeCurve": "curved armhole",
}
sleeve_cuff_shape_dict = {
    "CuffBand": "banded sleeve cuff",
    "CuffSkirt": "skirt sleeve cuff",
    "CuffBandSkirt": "banded skirt sleeve cuff",
    "null": "no sleeve cuff",
}

pants_cuff_shape_dict = {
    "CuffBand": "banded pants cuff",
    "CuffSkirt": "skirt pants cuff",
    "CuffBandSkirt": "banded skirt pants cuff",
    "null": "no pants cuff",
}
all_key_words = {}
for idx in tqdm.tqdm(range(len(gcd_folders))):
    pattern_folder = gcd_folders[idx]
    pattern_name = pattern_folder.split("/")[-1]
    design_param_file = glob.glob(os.path.join(pattern_folder, "*design_params.yaml"))
    if len(design_param_file) == 0:
        continue
    design_param_file = design_param_file[0]
    with open(design_param_file, "r") as f:
        design_params = yaml.load(f, Loader=yaml.FullLoader)
    design = design_params["design"]
    upper_name = design["meta"]["upper"]["v"]
    upper_name = upper_name_dict[upper_name] if upper_name else None
    fitted = "fitted" in upper_name if upper_name else False
    belt = design["meta"]["wb"]["v"]
    belt_name = belt_name_dict[belt] if belt else None
    lower_name = design["meta"]["bottom"]["v"]
    lower_name = lower_name_dict[lower_name] if lower_name else None
    if design["meta"]["bottom"]["v"] == "SkirtManyPanels":
        n_panels = design["flare-skirt"]["skirt-many-panels"]["n_panels"]["v"]
        lower_name = lower_name_dict["SkirtManyPanels"].format(n_panels=n_panels)
    if design["meta"]["bottom"]["v"] == "SkirtLevels":
        n_levels = design["levels-skirt"]["num_levels"]["v"]
        lower_name = lower_name_dict["SkirtLevels"].format(n_levels=n_levels)
    if design["meta"]["bottom"]["v"] == "GodetSkirt":
        n_panels = design["godet-skirt"]["num_inserts"]["v"]
        lower_name = lower_name_dict["GodetSkirt"].format(n_panels=n_panels)
    key_words = []
    if upper_name:
        key_words.append(upper_name)
    else:
        key_words.append("no top")
    if belt_name:
        key_words.append(belt_name)
    else:
        key_words.append("no waistband")
    if lower_name:
        key_words.append(lower_name)
    else:
        key_words.append("no bottom")
        
    if upper_name:
        if_assym = design["left"]["enable_asym"]["v"]
        
        if_strapless = design["shirt"]["strapless"]["v"] and fitted
        if if_strapless:
            key_words.append("strapless top")
        elif if_assym:
            key_words.append("asymmetric top")
            right_front_collar_type = design["collar"]["f_collar"]["v"]
            right_front_collar_type = collar_type_dict[right_front_collar_type]
            right_back_collar_type = design["collar"]["b_collar"]["v"]
            right_back_collar_type = collar_type_dict[right_back_collar_type]
            left_front_collar_type = design["left"]["collar"]["f_collar"]["v"]
            left_front_collar_type = collar_type_dict[left_front_collar_type]
            left_back_collar_type = design["left"]["collar"]["b_collar"]["v"]
            left_back_collar_type = collar_type_dict[left_back_collar_type]
            if right_front_collar_type == left_front_collar_type:
                key_words.append("front " + right_front_collar_type)
            else:
                key_words.append("front " + right_front_collar_type.replace("collar", "and ") + left_front_collar_type)
            if right_back_collar_type == left_back_collar_type:
                key_words.append("back " + right_back_collar_type)
            else:
                key_words.append("back " + right_back_collar_type.replace("collar", "and ") + left_back_collar_type)

            if_right_sleeveless = design["sleeve"]["sleeveless"]["v"]
            if_left_sleeveless = design["left"]["sleeve"]["sleeveless"]["v"]
            if if_right_sleeveless and if_left_sleeveless:
                key_words.append("sleeveless")
            elif if_right_sleeveless:
                key_words.append("no right sleeve")
                key_words.append("with left sleeve")
            elif if_left_sleeveless:
                key_words.append("no left sleeve")
                key_words.append("with right sleeve")
            else:
                left_armhole_shape = design["left"]["sleeve"]["armhole_shape"]["v"] if design["left"]["sleeve"]["sleeveless"]["v"] else "ArmholeCurve"
                right_armhole_shape = design["sleeve"]["armhole_shape"]["v"] if design["sleeve"]["sleeveless"]["v"] else "ArmholeCurve"
                if left_armhole_shape == right_armhole_shape:
                    key_words.append(armhole_shape_dict[left_armhole_shape])
                else:
                    key_words.append("left " + armhole_shape_dict[left_armhole_shape] + " and right " + armhole_shape_dict[right_armhole_shape])
                left_cuff_shape = design["left"]["sleeve"]["cuff"]["type"]["v"]
                if left_cuff_shape is None:
                    left_cuff_shape = "null"
                right_cuff_shape = design["sleeve"]["cuff"]["type"]["v"]
                if right_cuff_shape is None:
                    right_cuff_shape = "null"
                if left_cuff_shape == right_cuff_shape:
                    key_words.append(sleeve_cuff_shape_dict[left_cuff_shape])
                else:
                    key_words.append("left " + sleeve_cuff_shape_dict[left_cuff_shape] + " and right " + sleeve_cuff_shape_dict[right_cuff_shape])
        else:
            key_words.append("symmetric top")
            collar_component = design["collar"]["component"]["style"]["v"]
            collar_front_type = design["collar"]["f_collar"]["v"]
            collar_back_type = design["collar"]["b_collar"]["v"]
            if collar_component:
                collar_component = collar_comp_dict[collar_component]
                key_words.append(collar_component)
                collar_front_type = "CircleNeckHalf" if collar_component != "lapel neck" else collar_front_type
                collar_back_type = "CircleNeckHalf"
            collar_front_type = collar_type_dict[collar_front_type]
            collar_back_type = collar_type_dict[collar_back_type]
            if collar_front_type == collar_back_type:
                key_words.append(collar_front_type)
            else:
                key_words.append("front " + collar_front_type)
                key_words.append("back " + collar_back_type)
            
            armhole_shape = design["sleeve"]["armhole_shape"]["v"] if design["sleeve"]["sleeveless"]["v"] else "ArmholeCurve"
            armhole_shape = armhole_shape_dict[armhole_shape]
            key_words.append(armhole_shape)
            if_sleeveless = design["sleeve"]["sleeveless"]["v"]
            if if_sleeveless:
                key_words.append("sleeveless")
            else:
                key_words.append("with sleeve")
                cuff_shape = design["sleeve"]["cuff"]["type"]["v"]
                if cuff_shape is None:
                    cuff_shape = "null"
                cuff_shape = sleeve_cuff_shape_dict[cuff_shape]
                key_words.append(cuff_shape)
            
        



    if lower_name:
        if lower_name == "pants":
            
            cuff_shape = design["pants"]["cuff"]["type"]["v"]
            if cuff_shape is None:
                cuff_shape = "null"
            cuff_shape = pants_cuff_shape_dict[cuff_shape]
            key_words.append(cuff_shape)
        if "godet" in lower_name:
            base_skirt_name = design["godet-skirt"]["base"]["v"]
            if base_skirt_name == "PencilSkirt":
                key_words.append("pencil base skirt")
            elif base_skirt_name == "Skirt2":
                key_words.append("2-panel base skirt")
            n_panels = design["godet-skirt"]["num_inserts"]["v"]
            key_words.append(f"{n_panels} insets in godet skirt")
        if "tiered" in lower_name:
            base_skirt_name = design["levels-skirt"]["base"]["v"]
            if base_skirt_name == "PencilSkirt":
                key_words.append("pencil base skirt")
            elif base_skirt_name == "Skirt2":
                key_words.append("2-panel base skirt")
            elif base_skirt_name == "SkirtCircle":
                key_words.append("circle base skirt")
            elif base_skirt_name == "AsymmSkirtCircle":
                key_words.append("asymmetric circle base skirt")
            n_levels = design["levels-skirt"]["num_levels"]["v"]
            key_words.append(f"{n_levels} levels in tiered skirt")
        if "flare" in lower_name:
            n_panels = design["flare-skirt"]["skirt-many-panels"]["n_panels"]["v"]
            key_words.append(f"{n_panels} panels in flare skirt")
            
    all_key_words[pattern_name] = key_words
    
json.dump(all_key_words, open("/orion/u/w4756677/garment/gcdv2/short_captions_v2.json", "w"))
# json.dump(all_key_words, open("/orion/u/w4756677/garment/gcdv2/test_captions.json", "w"))
            


