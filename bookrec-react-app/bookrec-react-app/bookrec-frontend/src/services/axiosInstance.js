import  axios from "axios";

const baseUrl = 'http://127.0.0.1:5001/ml';
console.log(baseUrl)



const axiosInstance =axios.create({
    baseURL:baseUrl,
})


export default axiosInstance;