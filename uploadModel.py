import pickle
import numpy as np
from configparser import ConfigParser
import os
from tqdm import tqdm
import firebase_admin
from firebase_admin import firestore
from firebase_admin.firestore import firestore as fst

CONFIG_FILE = "config.ini"
SECTION = "options"
INPUT_FILE_OPTION = "input-file"
SURVEY_ID = "q93MdJh38JqPlb3tfWp8"


class TreeNode:
    id = ""

    is_root = False
    is_leaf = False

    question_title = ""
    threshold = 0

    child_left = ""
    child_right = ""

    result = ""


def finish():
    os.system("pause")
    exit()


def init_config(path):
    config = ConfigParser()
    config.read(path)

    if not config.has_section(SECTION):
        config.add_section(SECTION)

    return config


def save_config_file(config: ConfigParser):
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)


def is_yes(answer: str):
    if len(answer) == 0:
        return False

    return answer.strip().lower()[0] == 'y'


def file_exists(path: str):
    return os.path.isfile(path)


def ask_for_input_file():
    found = False
    path = ""

    while not found:
        path = input("Introduce el nombre del archivo que contiene el clasificador: ")

        if file_exists(path):
            found = True
        else:
            print(f"No se ha encontrado un archivo en \"{path}\"")

    return path


def get_input_file_path(config: ConfigParser):
    if config.has_option(SECTION, INPUT_FILE_OPTION):
        existing_file = config.get(SECTION, INPUT_FILE_OPTION)
        answer = input(f"¿Continuar con el archivo \"{existing_file}\"? (y/n): ")

        if is_yes(answer):
            if file_exists(existing_file):
                return existing_file
            else:
                print(f"No se ha encontrado un archivo en \"{existing_file}\"")

    input_file = ask_for_input_file()

    config.set(SECTION, INPUT_FILE_OPTION, input_file)
    save_config_file(config)

    return input_file


def parse_input_file(input_file):
    clf = pickle.load(open(input_file, "rb"))

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=values[i]
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=values[i],
                )
            )

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    feature_names = clf.feature_names_in_

    print(feature_names)

    print("Leyendo nodos del arbol...")

    nodes: list[TreeNode] = []

    for node_id in tqdm(range(n_nodes)):
        node = TreeNode()
        node.id = str(node_id)
        node.is_root = node_id == 0

        index_of_max_class = np.argmax(values[node_id])
        node.result = str(index_of_max_class)

        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            node.threshold = threshold[node_id] - 1
            feature_id = feature[node_id]
            node.question_title = feature_names[feature_id]

            left_child = children_left[node_id]
            right_child = children_right[node_id]
            
            node.child_left = str(left_child)
            node.child_right = str(right_child)
        else:
            node.is_leaf = True

        nodes.append(node)

    print("Completado!")

    return nodes


def connect_to_db():
    cred = firebase_admin.credentials.Certificate("surveycreator-ab3fb-firebase-adminsdk-7lf3z-c0b166bff7.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()


def delete_collection(coll_ref, batch_size):
    if batch_size == 0:
        return

    docs = coll_ref.list_documents(page_size=batch_size)
    deleted = 0

    for doc in docs:
        doc.delete()
        deleted = deleted + 1

    if deleted >= batch_size:
        return delete_collection(coll_ref, batch_size)


def remove_previous_nodes(db: fst.Client):
    answer = input("Se eliminaran los nodos que existian anteriormente ¿Continuar? (y/n) ")

    if is_yes(answer):
        print("Eliminando nodos existentes...")

        coll_ref = db.collection(f"Survey/{SURVEY_ID}/Node")

        delete_collection(coll_ref, 50)

        print("Completado!")
    else:
        finish()


cache = {}

def get_question_for_node(node: TreeNode, db: fst.Client):
    if node.id in cache:
        return cache[node.id]

    internal_question_title = ".".join(node.question_title.split(".")[:2]).upper()

    docs = (
        db.collection(f"Survey/{SURVEY_ID}/Question")
        .where(filter=fst.FieldFilter("InternalTitle", "==", internal_question_title))
        .stream()
    )

    for doc in docs:
        data = doc.to_dict()
        if data is not None:
            data["id"] = doc.id
            cache[node.id] = data
            return data

    print(f"Couldn't find question {internal_question_title}")
    finish()


def create_nodes(nodes: list[TreeNode], db: fst.Client):
    print("Almacenando nodos en la base de datos...")

    for node in tqdm(nodes):
        doc_ref = db.collection(f"Survey/{SURVEY_ID}/Node").document(node.id)

        if node.is_leaf:
            doc_ref.set({
                "IsRoot": node.is_root,
                "Result": node.result,
                "NextPerAnswer": {}
            })
        else:
            question = get_question_for_node(node, db)
            num_answers = len(question["DefaultDetails"]["Answers"])

            next_per_answer = {}
            for i in range(num_answers):
                if i <= node.threshold:
                    next_per_answer[str(i)] = node.child_left
                else:
                    next_per_answer[str(i)] = node.child_right

                

            doc_ref.set({
                "IsRoot": node.is_root,
                "Result": node.result,
                "NextPerAnswer": next_per_answer,
                "QuestionId": question["id"]
            })

    print("Completado!")


def get_load_order(nodes: list[TreeNode], db: fst.Client):
    print("Calculando el orden de cargado de las preguntas...")
    load_order: set[str] = set()

    stack: list[TreeNode] = []
    node_per_id = {}

    for node in nodes:
        if node.is_root and not node.is_leaf:
            stack.append(node)
        
        node_per_id[node.id] = node

    while len(stack) > 0:
        node = stack.pop(0)

        question = get_question_for_node(node, db)
        load_order.add(question["id"])

        left_child = node_per_id[node.child_left]
        right_child = node_per_id[node.child_right]
        
        if not left_child.is_leaf:
            stack.append(left_child)

        if not right_child.is_leaf:
            stack.append(right_child)

    print("Completado!")

    return load_order


def set_load_order(load_order: set[str], db: fst.Client):
    print("Actualizando el orden de cargado de la encuesta...")
    doc_ref = db.collection("Survey").document(SURVEY_ID)

    doc_ref.update({
        "LoadOrder": list(load_order)
    })
    print("Completado!")


def update_survey_load_order(nodes, db):
    load_order = get_load_order(nodes, db)

    print("\n")

    set_load_order(load_order, db)


def save_nodes_to_db(nodes: list[TreeNode]):
    db = connect_to_db()

    remove_previous_nodes(db)

    print("\n")

    create_nodes(nodes, db)

    print("\n")

    update_survey_load_order(nodes, db)


if __name__ == "__main__":
    config = init_config(CONFIG_FILE)

    input_file = get_input_file_path(config)

    print("\n")

    nodes = parse_input_file(input_file)

    print("\n")

    save_nodes_to_db(nodes)
