import React from "react";
import HeroSection from "../components/HeroSection";
import InfoSection from "../components/InfoSection";
import { homeObjOne } from "../components/InfoSection/Data";
const Home = () => {

  return (
    <>
      <HeroSection />
      <InfoSection {...homeObjOne}/>    
    </>
  );
};

export default Home;
