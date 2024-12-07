import React, { useState } from "react";
// import { herosectionbg2 as Image } from "../../common/imageUrls";
import Image from "../../common/assets/bgImage.jpg";
import {
  HeroContainer,
  HeroBg,
  ImageBg,
  HeroContent,
  HeroH1,
  HeroP,
  HeroBtnWrapper,
  ArrowForward,
  ArrowRight,
} from "./HeroElements";
import { Button } from "../ButtonElement";
import Navbar2 from "../../components/Navbar/index";
const HeroSection = () => {
  const [hover, setHover] = useState(false);

  const onHover = () => {
    setHover(!hover);
  };
  return (
    <>
    <Navbar2 />
    <HeroContainer>
      <HeroBg>
        <ImageBg src={Image} />
      </HeroBg>
      <HeroContent>
        <HeroH1>Book Recommendation System</HeroH1>
        <HeroP>
          <br />
          A Book Recommendation System using machine learning employs algorithms to suggest books to users based on their preferences, reading history, or similarity with other users or books. By utilizing techniques such as collaborative filtering, content-based filtering, or hybrid methods, it enhances personalized reading experiences.
           </HeroP>
        <HeroBtnWrapper>
          <Button
            to="products"
            onMouseEnter={onHover}
            onMouseLeave={onHover}
            primary="true"
            dark="true"
          >
            Get Started {hover ? <ArrowForward /> : <ArrowRight />}
          </Button>
        </HeroBtnWrapper>
      </HeroContent>
    </HeroContainer>
    </>
  );
};

export default HeroSection;
