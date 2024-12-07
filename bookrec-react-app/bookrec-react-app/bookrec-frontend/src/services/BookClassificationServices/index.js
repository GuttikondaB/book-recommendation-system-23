
import axiosInstance from "../axiosInstance";
const BOOK_API_URL = 'http://127.0.0.1:5001/ml';

class BookClassificationServices {

    getClusterRecommendations(userId) {
        return axiosInstance.get(`${BOOK_API_URL}/cluster_recommendations/${userId}`)
    }
    
    getSimilarBooks(bookTitle) {
        return axiosInstance.get(`${BOOK_API_URL}/similar_books/${bookTitle}`)
    }
    

}

export default new BookClassificationServices();
