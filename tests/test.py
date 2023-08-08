from word_embeddings_sdk import WordEmbeddingsSession

# Get these values from onboarding through website
customer_id = ""
api_key = ""
# where the API is located
host_url = ""

# set up your session and create a new model
session = WordEmbeddingsSession(customer_id=customer_id, api_key=api_key, url=host_url)
create_model_resp = session.create_model(model_name="testing model")

# get model info
model_info = session.get_models()['models'][0]

# kick off the finetuning job
finetune_info = session.finetune(
    model_id=model_info.get("id"),
    datasets=[{
        "loss": "TripletLoss",
        "loss_params": {
            "distance": "cosine",
            "margin": 0.5
        },
        "examples": [
            {"texts": ["Cheeseburger", "Hamburger", "Pizza"]},
            {"texts": ["Sushi", "Maki Roll", "Ice Cream"]},
            {"texts": ["Pancakes", "Waffles", "Salad"]},
            {"texts": ["Steak", "Ribeye", "Hot Dog"]},
            {"texts": ["Chicken Wings", "Buffalo Wings", "French Fries"]},
            {"texts": ["Tacos", "Burritos", "Nachos"]},
            {"texts": ["Spaghetti", "Lasagna", "Garlic Bread"]},
            {"texts": ["Sashimi", "Nigiri", "Tempura"]},
            {"texts": ["Donuts", "Cupcakes", "Muffins"]},
            {"texts": ["Pho", "Ramen", "Spring Rolls"]},
            {"texts": ["Fish and Chips", "Clam Chowder", "Onion Rings"]},
            {"texts": ["Fried Chicken", "Chicken Nuggets", "Mashed Potatoes"]},
            {"texts": ["Sushi Rolls", "California Roll", "Edamame"]},
            {"texts": ["Pasta Carbonara", "Fettuccine Alfredo", "Caesar Salad"]},
            {"texts": ["Gyoza", "Dumplings", "Fried Rice"]},
            {"texts": ["Cheesecake", "Brownies", "Creme Brulee"]},
            {"texts": ["Pad Thai", "Tom Yum Soup", "Thai Curry"]},
            {"texts": ["Fish Tacos", "Shrimp Tacos", "Guacamole"]},
            {"texts": ["Chicken Parmesan", "Meatball Subs", "Garlic Knots"]},
            {"texts": ["Burger and Fries", "Fish Sandwiches", "Onion Rings"]},
            {"texts": ["Tiramisu", "Cannoli", "Gelato"]},
            {"texts": ["Chicken Caesar Wrap", "Greek Salad", "Hummus"]},
            {"texts": ["Beef Stir Fry", "Sweet and Sour Chicken", "Egg Rolls"]},
            {"texts": ["Peking Duck", "Mongolian Beef", "Fried Rice"]},
            {"texts": ["Shrimp Scampi", "Lobster Bisque", "Crab Cakes"]},
            {"texts": ["Chicken Tikka Masala", "Naan Bread", "Samosas"]},
            {"texts": ["Fish and Chips", "Clam Chowder", "Onion Rings"]},
            {"texts": ["potato salad", "mashed potatoes", "sushi rolls"]},
            {"texts": ["cheeseburger", "hamburger", "ice cream"]},
            {"texts": ["steak", "ribeye steak", "salmon"]},
            {"texts": ["fried chicken", "grilled chicken", "lobster"]},
            {"texts": ["cheese", "mozzarella cheese", "chocolate"]},
            {"texts": ["sushi", "sashimi", "tempura"]},
            {"texts": ["fried rice", "steamed rice", "fried noodles"]}
        ]
    }]
)

# check on the finetuning job status
assert session.monitor_finetuning(finetune_info.get("finetune_id")), "Finetuning failed"

# get all model versions
model_versions = session.get_model_versions()

# model is done training, lets get some embeddings
embeddings = session.inference(model_id=model_info.get("id"), model_version_id=finetune_info.get("model_version_id"),
                               input_texts=["oatmeal cookie", "bagel", "fried chicken"])

# if you want to keep the resource that is hosting your model hot, you can use this function
session.keep_alive(model_id=model_info.get("id"), model_version_id=finetune_info.get("model_version_id"))

# your inferences will have a shorter delay between the call and response since you don't have to wait for the underlying resources to spin up
embeddings = session.inference(model_id=model_info.get("id"), model_version_id=finetune_info.get("model_version_id"),
                               input_texts=["shrimp poboy", "candy cane"])

# and then when you are finished running your inferences make sure to tear down your resources
if input("tear down stack? [Y/n] ") == "Y":
    session.tear_down(model_id=model_info.get("id"), model_version_id=finetune_info.get("model_version_id"))

# if you want to delete a model version
if input("delete model version? [Y/n] ") == "Y":
    session.delete_model_version(model_id=model_info.get("id"), model_version_id=finetune_info.get("model_version_id"))

# and if you want to delete a model
if input("delete model? [Y/n] ") == "Y":
    session.delete_model(model_id=model_info.get("id"))
