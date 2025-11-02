from KnowLedgeBased import option_choosen, random_option, show_all_options
from ContentBased import recommender_CB
from Collaborative import displayCF_New
from Hybrid import direct_hybrid_recommender
from BaaderMeinhof import baader_meinhof_engine
import warnings
warnings.filterwarnings('ignore')

import requests
import random
from typing import List, Dict
import time

class SystemMode:
    VERBOSE = "verbose"      # Hiển thị đầy đủ thông tin
    STATS_ONLY = "stats"     # Chỉ hiện kết quả và thống kê

# Biến toàn cục để lưu mode
current_mode = SystemMode.VERBOSE

def set_system_mode():
    """Chọn chế độ hiển thị cho hệ thống"""
    global current_mode
    print("\n🎛️  SYSTEM DISPLAY MODE")
    print("1. 📊 Verbose Mode (Hiển thị đầy đủ chi tiết)")
    print("2. 📈 Stats Only Mode (Chỉ hiện kết quả và thống kê)")
    
    choice = input("Chọn chế độ hiển thị (1/2, Default: 1): ").strip()
    
    if choice == "2":
        current_mode = SystemMode.STATS_ONLY
        print("🔕 Đã chọn Stats Only Mode - Chỉ hiện kết quả cuối cùng")
    else:
        current_mode = SystemMode.VERBOSE
        print("🔊 Đã chọn Verbose Mode - Hiển thị đầy đủ chi tiết")
    
    return current_mode

def log_info(message):
    """Chỉ log khi ở verbose mode"""
    if current_mode == SystemMode.VERBOSE:
        print(f"ℹ️  {message}")

def log_debug(message):
    """Chỉ log debug khi ở verbose mode"""
    if current_mode == SystemMode.VERBOSE:
        print(f"🔍 {message}")

def log_success(message):
    """Luôn hiển thị thành công"""
    print(f"✅ {message}")

def log_warning(message):
    """Luôn hiển thị cảnh báo"""
    print(f"⚠️  {message}")

def log_error(message):
    """Luôn hiển thị lỗi"""
    print(f"❌ {message}")

def display_stats_only(results, system_type):
    """Chỉ hiển thị kết quả và thống kê - FIXED VERSION"""
    print(f"\n📊 KẾT QUẢ {system_type.upper()} - STATS MODE")
    print("=" * 60)
    
    if system_type == "rtb":
        # Thống kê RTB - FIXED: Đếm bids chính xác
        successful_auctions = len([r for r in results if r.get('winning_bid')])
        
        # FIX: Đếm bids từ real_rtb_result thay vì all_bids_count
        total_bids = 0
        total_revenue = 0
        
        for r in results:
            if r.get('winning_bid'):
                # Lấy revenue từ winning bid
                total_revenue += r.get('winning_bid', {}).get('bid_price', 0)
            
            # Đếm bids từ real_rtb_result
            real_result = r.get('real_rtb_result', {})
            if real_result and 'all_bids' in real_result:
                total_bids += len(real_result['all_bids'])
            else:
                # Fallback: mỗi item có 3 bidding services
                total_bids += 3
        
        print(f"🎯 Successful Auctions: {successful_auctions}/{len(results)}")
        print(f"💰 Total Revenue: ${total_revenue:.2f}")
        print(f"📨 Total Bids: {total_bids}")
        print(f"📈 Avg Bids per Item: {total_bids/len(results):.1f}")
        
        # Top 3 auctions
        print(f"\n🏆 TOP 3 AUCTIONS:")
        top_auctions = sorted([r for r in results if r.get('winning_bid')], 
                            key=lambda x: x.get('winning_bid', {}).get('bid_price', 0), 
                            reverse=True)[:3]
        
        for i, auction in enumerate(top_auctions, 1):
            movie = auction.get('movie', 'Unknown')[:35]
            price = auction.get('winning_bid', {}).get('bid_price', 0)
            print(f"   {i}. {movie} | ${price:.2f}")
    
    elif system_type == "recommendation":
        # Thống kê recommendation
        if hasattr(results, 'shape'):  # DataFrame
            print(f"📋 Total Recommendations: {len(results)}")
            
            # Kiểm tra xem có cột final_score không
            if 'final_score' in results.columns:
                print(f"🎭 Score Range: {results['final_score'].min():.3f} - {results['final_score'].max():.3f}")
            
            # Phân loại recommendations nếu có
            if 'rec_type' in results.columns:
                hybrid_count = len(results[results['rec_type'] == "Hybrid"])
                cb_only_count = len(results[results['rec_type'] == "CB-Only"])
                cf_only_count = len(results[results['rec_type'] == "CF-Only"])
                print(f"🔀 Hybrid: {hybrid_count} | CB-Only: {cb_only_count} | CF-Only: {cf_only_count}")
            
            # Top 5 recommendations
            print(f"\n🏅 TOP 5 RECOMMENDATIONS:")
            display_cols = ['title']
            if 'final_score' in results.columns:
                display_cols.append('final_score')
            if 'rec_type' in results.columns:
                display_cols.append('rec_type')
                
            for i, (_, row) in enumerate(results.head(5).iterrows(), 1):
                title_display = row['title'][:35] if len(row['title']) > 35 else row['title']
                score_display = f" | Score: {row['final_score']:.3f}" if 'final_score' in row else ""
                type_display = f" | {row['rec_type']}" if 'rec_type' in row else ""
                print(f"   {i}. {title_display}{score_display}{type_display}")
        else:
            # List recommendations
            print(f"📋 Total Recommendations: {len(results)}")
            print(f"\n🏅 TOP RECOMMENDATIONS:")
            for i, movie in enumerate(results[:5], 1):
                movie_display = movie[:35] if len(movie) > 35 else movie
                print(f"   {i}. {movie_display}")

class RealRTBClient:   
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def send_real_rtb_request(self, user_context: Dict, item_context: Dict, final_score: float = None) -> Dict:
        try:
            request_data = {
                "user_context": user_context,
                "item_context": item_context
            }
            
            if final_score is not None:
                request_data["final_score"] = final_score
            
            response = self.session.post(
                f"{self.base_url}/rtb/bid-enhanced",
                json=request_data,
                timeout=5.0
            )
            return response.json()
        except Exception as e:
            log_error(f"Real RTB failed: {e}, falling back to mock")
            return self.fallback_mock_request(user_context, item_context, final_score)
    
    def fallback_mock_request(self, user_context: Dict, item_context: Dict, final_score: float = None) -> Dict:
        mock_service = RTBMockService()
        return mock_service.mock_bid_request(user_context, item_context, final_score)

class RTBMockService:
    def __init__(self):
        self.bidding_services = 3
        
    def mock_bid_request(self, user_context: Dict, item_context: Dict, final_score: float = None) -> Dict:
        time.sleep(0.02)
        
        # Tính match_score trước khi sử dụng
        user_prefs = user_context.get("preferences", [])
        item_genres = item_context.get("targeting_criteria", {}).get("genres", ["drama"])
        match_score = len(set(user_prefs) & set(item_genres)) / max(len(item_genres), 1)
        
        if final_score is not None:
            base_multiplier = final_score * 3.0 + 0.5
        else:
            base_multiplier = match_score * 2.0 + 0.5
            
        all_bids = []
        for i in range(self.bidding_services):
            service_multiplier = random.uniform(0.8, 1.3)
            base_price = item_context.get("base_price", 2.5)
            bid_price = base_price * base_multiplier * service_multiplier
            bid_price = max(1.0, min(bid_price, 15.0))
            
            all_bids.append({
                "service_id": i + 1,
                "bid_price": round(bid_price, 2),
                "creative_id": f"creative_{random.randint(1000, 9999)}",
                "final_score_used": final_score,
                "base_multiplier": round(base_multiplier, 2),
                "timestamp": time.time()
            })
            
        # Determine winning bid (second-price auction)
        sorted_bids = sorted(all_bids, key=lambda x: x["bid_price"], reverse=True)
        winning_bid = sorted_bids[0] if sorted_bids else None
        
        if winning_bid and len(sorted_bids) >= 2:
            winning_bid["winning_price"] = sorted_bids[1]["bid_price"]
        elif winning_bid:
            winning_bid["winning_price"] = winning_bid["bid_price"]
        
        return {
            "auction_id": f"auction_{int(time.time())}_{random.randint(1000, 9999)}",
            "timestamp": time.time(),
            "winning_bid": winning_bid,
            "all_bids": all_bids,
            "total_services_contacted": self.bidding_services,
            "successful_services": len(all_bids),
            "service_success_rate": len(all_bids) / self.bidding_services,
            "match_score": match_score
        }

class RTBTestClient:
    def __init__(self, base_url="http://localhost:8000", use_mock=True):
        self.base_url = base_url
        self.use_mock = use_mock
        self.users = [
            {"user_id": "user_001", "session_id": "sess_001", "age": 25, "gender": "M", "preferences": ["action", "drama", "crime"]},
            {"user_id": "user_002", "session_id": "sess_002", "age": 30, "gender": "F", "preferences": ["comedy", "romance", "drama"]},
            {"user_id": "user_003", "session_id": "sess_003", "age": 35, "gender": "M", "preferences": ["thriller", "crime", "action"]}
        ]
        self.mock_service = RTBMockService()
    
    def send_rtb_request(self, user_context: Dict, item_context: Dict) -> Dict:
        return self.mock_service.mock_bid_request(user_context, item_context)
    
    def simulate_rtb_for_recommendations(self, recommendations: List[str], user_index: int = 0) -> List[Dict]:
        results = []
        user = self.users[user_index]
        
        if current_mode == SystemMode.VERBOSE:
            print(f"\nSIMULATING RTB BIDDING FOR USER: {user['user_id']}")
            print("=" * 60)
        
        test_recommendations = recommendations[:30]
        
        for i, movie_title in enumerate(test_recommendations):
            item_context = self.create_item_context(movie_title, user)
            
            # Gọi mock service
            result = self.send_rtb_request(user, item_context)
            
            bid_result = {
                "rank": i + 1,
                "movie": movie_title,
                "auction_id": result.get("auction_id", "N/A"),
                "winning_bid": result.get("winning_bid", {}),
                "all_bids_count": len(result.get("all_bids", [])),
                "success_rate": result.get("service_success_rate", 0),
                "match_score": result.get("match_score", 0)
            }
            
            results.append(bid_result)
            
            # Chỉ hiển thị real-time khi ở verbose mode
            if current_mode == SystemMode.VERBOSE:
                winning_bid = bid_result["winning_bid"]
                if winning_bid:
                    winning_price = winning_bid.get("bid_price", 0)
                    winning_price_display = winning_bid.get("winning_price", winning_price)
                    bid_count = bid_result["all_bids_count"]
                    match_score = bid_result["match_score"]
                    status = "✅"
                    
                    print(f"{status} {i+1:2d}. {movie_title:<35} | Winning: ${winning_price_display:.2f} | Bids: {bid_count} | Match: {match_score:.1%}")
                else:
                    print(f"❌ {i+1:2d}. {movie_title:<35} | No winning bid | Bids: 0")
        
        return results

    def display_decayed_results(self, decayed_results: List[Dict]):
        """Hiển thị kết quả sau khi apply Baader-Meinhof decay"""
        if current_mode != SystemMode.VERBOSE:
            return
            
        print(f"\n🎭 BAADER-MEINHOF DECAY RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(decayed_results[:10], 1):
            movie = result['movie']
            original_score = result.get('original_score', result.get('final_score', 0))
            decayed_score = result.get('final_score', 0)
            decay_multiplier = result.get('baader_meinhof_decay', 1.0)
            
            score_change = decayed_score - original_score
            change_indicator = "🔻" if score_change < 0 else "🔺" if score_change > 0 else "➡️"
            
            print(f"{i:2d}. {movie:<35} | "
                  f"Score: {decayed_score:.3f} ({change_indicator}{abs(score_change):.3f}) | "
                  f"Decay: {decay_multiplier:.2f}x")
            
            # Hiển thị additional info cho debug
            if decay_multiplier < 0.7:
                print(f"   ⚠️  Low decay due to recent exposure/frequency")

    def simulate_real_rtb_for_recommendations_with_scores(self, recommendations_with_scores: List[Dict], user_index: int = 0, searched_movie: str = None) -> List[Dict]:
        results = []
        user = self.users[user_index]
        
        if current_mode == SystemMode.VERBOSE:
            print(f"\n REAL RTB BIDDING FOR USER: {user['user_id']}")
            print("=" * 60)
            print(" Connecting to Distributed RTB Cluster...")
        
        real_client = RealRTBClient()
        
        for i, movie_data in enumerate(recommendations_with_scores[:12]):
            movie_title = movie_data['title']
            final_score = movie_data['final_score']
            
            item_context = self.create_item_context(movie_title, user, final_score)
            
            result = real_client.send_real_rtb_request(user, item_context, final_score)
            
            bid_result = self.process_real_rtb_response(result, i, movie_title, final_score)
            results.append(bid_result)
        
        # 🚀 APPLY BAADER-MEINHOFF DECAY WITH SEARCH BOOST
        if current_mode == SystemMode.VERBOSE:
            print(f"\n🎭 APPLYING BAADER-MEINHOFF DECAY WITH SEARCH BOOST...")
        
        decayed_results = baader_meinhof_engine.apply_baader_meinhof_decay(
            user['user_id'], 
            results,
            searched_movie
        )
        
        # Update exposure history
        movie_titles = [result['movie'] for result in decayed_results]
        baader_meinhof_engine.update_exposure_history(user['user_id'], movie_titles)
        
        # Hiển thị kết quả sau adjustment
        if current_mode == SystemMode.VERBOSE:
            self.display_adjusted_results(decayed_results)
        
        return decayed_results

    def process_real_rtb_response(self, result: Dict, index: int, movie_title: str, final_score: float) -> Dict:
        if current_mode != SystemMode.VERBOSE:
            # Ở stats mode, chỉ trả về kết quả không hiển thị
            return {
                "rank": index + 1,
                "movie": movie_title,
                "final_score": final_score,
                "real_rtb_result": result,
                "consensus_achieved": result.get("consensus_metrics", {}).get("quorum_achieved", False),
                "consensus_level": result.get("consensus_metrics", {}).get("consensus_level", "NO_CONSENSUS"),
                "winning_bid": result.get("winning_bid")
            }
            
        winning_bid = result.get("winning_bid")
        consensus_metrics = result.get("consensus_metrics", {})
        
        quorum_achieved = consensus_metrics.get("quorum_achieved", False)
        quorum_actual = consensus_metrics.get("successful_responses", 0)
        quorum_required = consensus_metrics.get("total_services", 1)
        consensus_level = consensus_metrics.get("consensus_level", "NO_CONSENSUS")
        
        if winning_bid and quorum_achieved:
            winning_price = winning_bid.get("winning_price", 0)
            original_bid_price = winning_bid.get("bid_price", 0)
            second_price_source = winning_bid.get("second_price_source", "unknown")

            if winning_price != original_bid_price:
                auction_type = "SECOND-PRICE"
                price_display = f"Pay: ${winning_price:.2f} (2nd) | Bid: ${original_bid_price:.2f} (1st)"
            else:
                auction_type = "SINGLE-BID"
                price_display = f"Pay: ${winning_price:.2f} | Only 1 bidder"
            
            expected_price_range = f"${2.0 + final_score * 6.0:.1f}-${4.0 + final_score * 8.0:.1f}"
            
            print(f"{index+1:2d}. {movie_title:<35} | {auction_type}")
            print(f"   {price_display}")
            print(f"   Score: {final_score:.3f} | Expected: {expected_price_range}")
            print(f"   Consensus: {consensus_level} | Quorum: {quorum_actual}/{quorum_required}")
            
        else:
            print(f"{index+1:2d}. {movie_title:<35} | No winning bid | Score: {final_score:.3f}")
        
        return {
            "rank": index + 1,
            "movie": movie_title,
            "final_score": final_score,
            "real_rtb_result": result,
            "consensus_achieved": quorum_achieved,
            "consensus_level": consensus_level,
            "winning_bid": winning_bid
        }

    def create_item_context(self, movie_title: str, user_context: Dict, final_score: float = None) -> Dict:
        genre_map = {
            "godfather": ["crime", "drama"],
            "honor thy": ["drama", "family"], 
            "family": ["drama", "crime"],
            "blood ties": ["drama", "thriller"],
            "mother": ["drama", "family"],
            "election": ["drama", "comedy"],
            "outside man": ["action", "crime"],
            "johnny": ["comedy", "crime"],
            "milk": ["drama", "family"],
            "live by night": ["crime", "drama"],
            "plan": ["thriller", "action"],
            "marry": ["comedy", "romance"],
            "natsamrat": ["drama"],
            "christmas": ["family", "drama"],
            "cave": ["adventure", "family"],
            "friendship": ["drama"],
            "queen": ["drama", "romance"],
            "shanghai": ["drama", "crime"],
            "easy money": ["crime", "drama"],
            "outlaw": ["drama", "adventure"],
            "gang war": ["action", "crime"],
            "miss bala": ["action", "thriller"],
            "saints": ["drama"],
            "made": ["comedy", "drama"]
        }
        
        movie_lower = movie_title.lower()
        genres = ["drama"]
        
        for keyword, genre_list in genre_map.items():
            if keyword in movie_lower:
                genres = genre_list
                break
        base_price = 3.0 
        
        base_item = {
            "item_id": f"movie_{random.randint(1000, 9999)}",
            "item_type": "movie",
            "title": movie_title,
            "category": "entertainment",
            "base_price": 3.0,
            "final_score": final_score,
            "targeting_criteria": {
                "genres": genres,
                "min_age": 18,
                "languages": ["en"],
                "quality_tier": "premium" if any(keyword in movie_lower for keyword in ["godfather", "honor", "family"]) else "standard"
            }
        }
        return base_item

    def display_adjusted_results(self, adjusted_results: List[Dict]):
        """Hiển thị kết quả sau khi apply search boost + decay - FIXED VERSION"""
        if current_mode != SystemMode.VERBOSE:
            return
            
        print(f"\n🎭 SEARCH BOOST + DECAY RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(adjusted_results[:10], 1):
            movie = result.get('movie', 'Unknown')
            original_score = result.get('original_score', result.get('final_score', 0))
            adjusted_score = result.get('final_score', 0)
            multiplier = result.get('baader_meinhof_multiplier', 1.0)
            adjustment_type = result.get('adjustment_type', 'NEUTRAL')
            
            score_change = adjusted_score - original_score
            change_indicator = "🔻" if score_change < 0 else "🔺" if score_change > 0 else "➡️"
            
            adjustment_icon = "🚀" if adjustment_type == "BOOST" else "📉" if adjustment_type == "DECAY" else "➡️"
            
            print(f"{i:2d}. {movie:<35} | "
                  f"Score: {adjusted_score:.3f} ({change_indicator}{abs(score_change):.3f}) | "
                  f"Multiplier: {multiplier:.2f}x {adjustment_icon}")
            
            # Hiển thị additional info
            if multiplier > 1.2:
                print(f"   💫 Search boost applied!")
            elif multiplier < 0.7:
                print(f"   ⚠️  Low score due to recent exposure/frequency")

def display_real_distributed_hybrid():
    print("\n🎯 REAL DISTRIBUTED HYBRID + SEARCH BOOST ENGINE")
    
    user_input = input("Enter user ID (Default: 1): ").strip()
    movie_input = input("Enter movie name to SEARCH: ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    
    user_id = int(user_input) if user_input.isdigit() else 1
    top_n = int(top_input) if top_input.isdigit() else 10
    
    log_info(f"Searching for movies related to '{movie_input}'...")
    results = direct_hybrid_recommender.direct_hybrid(user_id, movie_input, top_n)
    
    if not results.empty:
        if current_mode == SystemMode.STATS_ONLY:
            display_stats_only(results, "recommendation")
        else:
            print(f"\nHYBRID RECOMMENDATIONS FOR USER {user_id}")
            print("=" * 70)
            print(results.to_string(index=False))
        
        # Chuyển recommendations thành format cho RTB
        recommendations_with_scores = []
        for _, row in results.iterrows():
            recommendations_with_scores.append({
                'title': row['title'],
                'final_score': row['final_score']
            })
        
        # Chạy RTB + Search Boost + Decay
        rtb_client = RTBTestClient()
        final_results = rtb_client.simulate_real_rtb_for_recommendations_with_scores(
            recommendations_with_scores, 
            user_index=0,
            searched_movie=movie_input
        )
        
        # Hiển thị statistics
        if current_mode == SystemMode.STATS_ONLY:
            display_stats_only(final_results, "rtb")
        else:
            stats = baader_meinhof_engine.get_user_exposure_stats(str(user_id))
            print(f"\n📊 SEARCH BOOST ENGINE STATS:")
            print(f"   - Total user exposures: {stats.get('total_exposures', 0)}")
            print(f"   - Recent exposures (6h): {stats.get('recent_exposures_6h', 0)}")
            print(f"   - Searched movies count: {stats.get('searched_movies_count', 0)}")
        
        return True
    else:
        log_error("No hybrid recommendations found!")
        return False
    
def option_Knowledge():
    print("\nKNOWLEDGE_BASED RECOMMENDATION")
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
        if examples and current_mode == SystemMode.VERBOSE:
            print(f"\nExample {filterVal} exist:")
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
        log_info(f"Searching from {filterVal}: {value}...")
        try:
            result = option_choosen(filterVal, value)
            if not result.empty:
                if current_mode == SystemMode.STATS_ONLY:
                    display_stats_only(result, "recommendation")
                else:
                    print(f"\nFound {len(result)} movies:")
                    display_columns = ['title', 'vote_average']
                    if 'wr' in result.columns:
                        display_columns.append('wr')
                    print(result[display_columns].head(top).to_string(index=False))
            else:
                log_warning("Not Found")
        except Exception as e:
            log_error(f"Error: {e}")

def Content_Based():
    print("\n🎬 CONTENT_BASED RECOMMENDATION")
    movie = input("Enter the movie's name (e.g., The Dark Knight): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    top = int(top_input) if top_input.isdigit() else 10
    return movie, top

def displayCB():
    """Content-Based với Stats Mode Support"""
    movie, top = Content_Based()
    if not movie:
        log_error("Movie name cannot be empty!")
        return False
    
    log_info(f"Searching for movies similar to '{movie}'...")
    try:
        result = recommender_CB(movie, top)
        if result and len(result) > 0:
            if current_mode == SystemMode.STATS_ONLY:
                display_stats_only(result, "recommendation")
            else:
                print(f"\nRecommended {len(result)} movies:")
                for i, title in enumerate(result, 1):
                    print(f"{i}. {title}")
            return True
        else:
            log_warning("No recommendations found!")
            return False
    except Exception as e:
        log_error(f"Error: {e}")
        return False

def Collaborative():
    print("\n👥 COLLABORATIVE RECOMMENDATION")
    user_input = input("Enter user ID (Default: 2): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    userId = int(user_input) if user_input.isdigit() else 2
    top = int(top_input) if top_input.isdigit() else 10
    return userId, top

def displayCF():
    displayCF_New()

def display_direct_hybrid():
    """Direct Hybrid với Stats Mode Support"""
    print("\nDIRECT HYBRID RECOMMENDATION SYSTEM")
    
    user_input = input("Enter user ID (Default: 1): ").strip()
    movie_input = input("Enter movie name: ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    
    user_id = int(user_input) if user_input.isdigit() else 1
    top_n = int(top_input) if top_input.isdigit() else 10
    
    log_info("Generating direct hybrid recommendations...")
    results = direct_hybrid_recommender.direct_hybrid(user_id, movie_input, top_n)
    
    if not results.empty:
        if current_mode == SystemMode.STATS_ONLY:
            display_stats_only(results, "recommendation")
        else:
            print(f"\nDIRECT HYBRID RECOMMENDATIONS FOR USER {user_id}")
            print("=" * 70)
            print(results.to_string(index=False))

        rtb_client = RTBTestClient()
        recommendations_list = results['title'].tolist()
        
        log_info(f"Starting RTB simulation for {len(recommendations_list)} recommendations")
        
        # Dùng SYNC version
        rtb_results = rtb_client.simulate_rtb_for_recommendations(recommendations_list)
        
        if current_mode == SystemMode.STATS_ONLY:
            display_stats_only(rtb_results, "rtb")
        else:
            # Hiển thị summary đầy đủ
            successful_auctions = len([r for r in rtb_results if r['winning_bid']])
            total_bids = sum([r['all_bids_count'] for r in rtb_results])
            total_revenue = sum([r['winning_bid'].get('bid_price', 0) for r in rtb_results if r['winning_bid']])
            
            print(f"\nRTB SIMULATION SUMMARY:")
            print(f"   - Successful auctions: {successful_auctions}/{len(rtb_results)}")
            print(f"   - Total bids placed: {total_bids}")
            print(f"   - Total revenue: ${total_revenue:.2f}")
        
        return True
    else:
        log_error("No direct hybrid recommendations found!")
        return False

def menu_option():
    """Menu chính với option chọn mode"""
    print("\n🎛️  RECOMMENDER SYSTEM + RTB TESTING")
    print("📊 SYSTEM MODE:")
    print("   1. Verbose Mode (Hiển thị đầy đủ)")
    print("   2. Stats Only Mode (Chỉ hiện kết quả)")
    print("\n🎯 RECOMMENDATION SYSTEMS:")
    print("   3. Knowledge-Based")
    print("   4. Content-Based") 
    print("   5. Collaborative") 
    print("   6. Hybrid (Direct) + RTB Simulation")
    print("   7. Hybrid + REAL Distributed RTB")
    print("   0. Exit")

    choice = input("Choose option (0-7): ").strip()
    return choice

def main():
    global current_mode
    
    while True:
        choice = menu_option()
        
        if choice == '1':
            current_mode = SystemMode.VERBOSE
            print("🔊 Đã chọn Verbose Mode - Hiển thị đầy đủ chi tiết")
            continue
        elif choice == '2':
            current_mode = SystemMode.STATS_ONLY
            print("🔕 Đã chọn Stats Only Mode - Chỉ hiện kết quả cuối cùng")
            continue
        elif choice == '3':
            displayKN()
        elif choice == '4':
            displayCB()
        elif choice == '5':
            displayCF()
        elif choice == '6':  
            display_direct_hybrid()   
        elif choice == '7':
            display_real_distributed_hybrid()
        elif choice == '0':
            print("\nThank for choosing our system - Bye :3")
            break
        else:
            print("Wrong choice!")
        
        if choice not in ['0', '1', '2']:
            reply = input("You want to continue? (y/n): ").strip().lower()
            if reply not in ['y', 'yes']:
                print("\nThank for choosing our system - Bye :3")
                break

if __name__ == "__main__":
    main()