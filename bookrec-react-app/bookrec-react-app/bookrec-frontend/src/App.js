import React from "react";
import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./pages";
import Footer from "./components/Footer";
import ClusterRecommendations from "./pages/ClusterRecommendations";
import SimilarBooks from "./pages/SimilarBooks";

function App() {
  return (
  <>
    <Router>
      <Routes>
        <Route path='/' element={<Home/>} exact/>
        <Route path='/bookrecommendationbycluster' element={<ClusterRecommendations/>} exact />
        <Route path='/similarbookrecommendation' element={<SimilarBooks/>} exact />
      </Routes>
    </Router>
    <Footer />
    </>
  );
};

export default App;
