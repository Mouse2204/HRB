from KnowLedgeBased import option_choosen, random_option, show_all_options
from ContentBased import recommender_CB
from Collaborative import displayCF_improved
from Hybrid import direct_hybrid_recommender
import warnings
warnings.filterwarnings('ignore')

def option_Knowledge():
    print("\n📚 KNOWLEDGE_BASED RECOMMENDATION")
    print("Option:")
    print("1. Genres_list")
    print("2. Keyword")
    print("3. Crew_name")
    print("4. Character")
    print("5. Example Option")
    print("0. Exit")
    number = input("Choose your filter(0-5) or (zero-five): ").strip().lower()

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
    elif number in ['5', 'five']:
        print("\nFilter")
        print("1. Genres_list")
        print("2. Keyword") 
        print("3. Crew_name")
        print("4. Character")
        choice = input("Choose your filter: ").strip()
        sub_type_map = {'1': 'genres_list', '2': 'keyw', '3': 'crew_name', '4': 'character'}
        if choice in sub_type_map:
            show_all_options(sub_type_map[choice])
        return None, None, None
    elif number in type_map:
        filterVal = type_map[number]
        examples = random_option(filterVal, 5)
        if examples:
            print(f"\n💡 Example {filterVal} exist:")
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example}")
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
    displayCF_improved()

def get_hybrid_recommendations():
    """
    Lấy input cho hybrid recommendations
    """
    print("\n🎯 HYBRID RECOMMENDATION SYSTEM")
    print("(Personalized recommendations based on user + movie content)")
    
    user_input = input("Enter user ID (Default: 1): ").strip()
    movie_input = input("Enter movie name (e.g., Avatar): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    
    user_id = int(user_input) if user_input.isdigit() else 1
    top_n = int(top_input) if top_input.isdigit() else 10
    
    # Cho phép điều chỉnh weights
    print("\n⚖️ Adjust recommendation weights:")
    cf_input = input("CF weight (user behavior) [0-1, Default: 0.5]: ").strip()
    cb_input = input("CB weight (content similarity) [0-1, Default: 0.5]: ").strip()
    
    cf_weight = float(cf_input) if cf_input.replace('.', '').isdigit() else 0.5
    cb_weight = float(cb_input) if cb_input.replace('.', '').isdigit() else 0.5
    
    # Normalize weights
    total = cf_weight + cb_weight
    cf_weight /= total
    cb_weight /= total
    
    return user_id, movie_input, top_n, cf_weight, cb_weight

def display_direct_hybrid():
    """Hiển thị direct hybrid recommendations"""
    print("\n🎯 DIRECT HYBRID RECOMMENDATION SYSTEM")
    print("(Combines CB similarity + CF predictions in one step)")
    
    user_input = input("Enter user ID (Default: 1): ").strip()
    movie_input = input("Enter movie name (e.g., Avatar): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    
    user_id = int(user_input) if user_input.isdigit() else 1
    top_n = int(top_input) if top_input.isdigit() else 10
    
    print(f"\n🔄 Generating direct hybrid recommendations...")
    results = direct_hybrid_recommender.direct_hybrid(user_id, movie_input, top_n)
    
    if not results.empty:
        print(f"\n🎯 DIRECT HYBRID RECOMMENDATIONS FOR USER {user_id}")
        print("=" * 70)
        print(results.to_string(index=False))
        return True
    else:
        print("❌ No direct hybrid recommendations found!")
        return False
def menu_option():
    print("\n🎬 RECOMMENDER SYSTEM")
    print("1. Knowledge-Based")
    print("2. Content-Based") 
    print("3. Collaborative") 
    print("4. Hybrid (Direct)")
    print("0. Exit")

    choice = input("Choose your system(0-4): ").strip()
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
        elif choice == '4':  
            display_direct_hybrid()   
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