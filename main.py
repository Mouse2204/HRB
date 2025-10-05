from KnowLedgeBased import option_choosen
from ContentBased import recommender_CB
from Collaborative import trainML, collaborative_filtering
import warnings
warnings.filterwarnings('ignore')

def option_Knowledge():
    print("\n📚 KNOWLEDGE_BASED RECOMMENDATION")
    print("Filter:")
    print("1. Genres_list")
    print("2. Keyword")
    print("3. Crew_name")
    print("4. Character")
    print("0. Exit")
    number = input("Choose your filter(0-4) or (zero-four): ").strip().lower()

    type_map = {
        '1': 'genres_list',
        'one': 'genres_list',
        '2': 'keyw',
        'two': 'keyw',
        '3': 'crew_name',
        'three': 'crew_name',
        '4': 'character',
        'four': 'character',
        '0': 'exit',
        'zero': 'exit'
    }

    if number in ['0', 'zero']:
        return None, None, None
    elif number in type_map:
        filterVal = type_map[number]
        value = input(f"Choose your favorite for {filterVal}: ").strip()
        top_input = input("Number of recommend movie (Default is 10): ").strip()
        top = int(top_input) if top_input.isdigit() else 10
        return filterVal, value, top
    else:
        print("Wrong choice - Error")
        return None, None, None
    
def displayKN():
    filterVal, value, top = option_Knowledge()
    if filterVal and value:
        print(f"\nWaiting for searching from {filterVal}: {value}...")
        try:
            result = option_choosen(filterVal, value)
            if not result.empty:
                print(f"\nFound {len(result)} movies:")
                display_columns = ['title', 'vote_average']
                if 'wr' in result.columns:
                    display_columns.append('wr')
                print(result[display_columns].head(top).to_string(index=False))
            else:
                print("Not Found")
        except Exception as e:
            print(f"Error: {e}")

def Content_Based():
    print("\n🎬 CONTENT_BASED RECOMMENDATION")
    movie = input("Enter the movie's name (e.g., The Dark Knight): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    top = int(top_input) if top_input.isdigit() else 10
    return movie, top

def displayCB():
    movie, top = Content_Based()
    if not movie:
        print("Movie name cannot be empty!")
        return False
    
    print(f"\nSearching for movies similar to '{movie}'...")
    try:
        result = recommender_CB(movie, top)
        if result and len(result) > 0:
            print(f"\nRecommended {len(result)} movies:")
            for i, title in enumerate(result, 1):
                print(f"{i}. {title}")
            return True
        else:
            print("No recommendations found!")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def Collaborative():
    print("\n👥 COLLABORATIVE RECOMMENDATION")
    user_input = input("Enter user ID (Default: 2): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    userId = int(user_input) if user_input.isdigit() else 2
    top = int(top_input) if top_input.isdigit() else 10
    return userId, top

def displayCF():
    userId, top = Collaborative()
    print(f"\nGenerating recommendations for user {userId}...")
    try:
        model_train = trainML()
        if model_train and len(model_train) == 2:
            model, rmse = model_train
            print(f"Model RMSE: {rmse:.4f}")
            results = collaborative_filtering(userId, model, top)
            if hasattr(results, 'empty') and not results.empty:
                print(f"\nRecommended {len(results)} movies:")
                print(results.to_string(index=False))
                return True
        print("No recommendations found!")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def menu_option():
    print("\nRECOMMENDER SYSTEM")
    print("1. Knowledge_Based")
    print("2. Content_Based")
    print("3. Collaborative")
    print("0. Exit")

    choice = input("Choose your system(0-3): ").strip()
    return choice

def main():
    while True:
        choice = menu_option()
        if choice == '1':
            displayKN()
        elif choice == '2':
            displayCB()
        elif choice == '3':
            displayCF()
        elif choice == '0':
            print("\nThank for choosing our system - Bye :3")
            break
        else:
            print("Wrong choice!")
        
        if choice != '0':
            reply = input("You want to continue? (y/n): ").strip().lower()
            if reply not in ['y', 'yes']:
                print("\nThank for choosing our system - Bye :3")
                break

if __name__ == "__main__":
    main()