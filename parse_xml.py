import xml.etree.ElementTree as ET

def parse_tracklet_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # Check the structure of the XML
    for tracklet in root.findall('.//item'):
        object_type = tracklet.find('objectType').text if tracklet.find('objectType') is not None else None
        if object_type != 'Pedestrian':
            continue
        
        first_frame = int(tracklet.find('first_frame').text) if tracklet.find('first_frame') is not None else None
        poses = tracklet.find('poses')
        
        if poses is not None:
            for pose in poses.findall('item'):
                tx = float(pose.find('tx').text) if pose.find('tx') is not None else None
                ty = float(pose.find('ty').text) if pose.find('ty') is not None else None
                tz = float(pose.find('tz').text) if pose.find('tz') is not None else None
                rx = float(pose.find('rx').text) if pose.find('rx') is not None else None
                ry = float(pose.find('ry').text) if pose.find('ry') is not None else None
                rz = float(pose.find('rz').text) if pose.find('rz') is not None else None
                state = int(pose.find('state').text) if pose.find('state') is not None else None
                occlusion = int(pose.find('occlusion').text) if pose.find('occlusion') is not None else None
                truncation = int(pose.find('truncation').text) if pose.find('truncation') is not None else None

                annotation = {
                    'type': object_type,
                    'first_frame': first_frame,
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                    'rx': rx,
                    'ry': ry,
                    'rz': rz,
                    'state': state,
                    'occlusion': occlusion,
                    'truncation': truncation
                }
                annotations.append(annotation)

    return annotations

xml_file = r'C:\Users\n2309064h\Desktop\Multimodal_code\kitti\2011_09_26\2011_09_26_drive_0005_sync\tracklet_labels.xml'
annotations = parse_tracklet_xml(xml_file)
print(f'Parsed {len(annotations)} annotations')
print(annotations[:5])  # Print the first 5 annotations to verify



#annotations = parse_tracklet_labels(r'C:\Users\n2309064h\Desktop\Multimodal_code\kitti\2011_09_26\2011_09_26_drive_0005_sync\tracklet_labels.xml')
