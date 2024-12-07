import React from "react";
import { animateScroll as scroll } from "react-scroll";
import { Nav, NavbarContainer, NavLogo } from "./NavbarElements";
import LogoTM from "../../common/assets/untlogo.png";

const Navbar2 = () => {
  const toggleHome = () => {
    scroll.scrollToTop()
  }
  return (
    <>
      <Nav>
        <NavbarContainer>
          <NavLogo onClick={toggleHome} to="/">
            <img src={LogoTM} alt="" />
          </NavLogo>
          <NavLogo onClick={toggleHome} to="/">
            Book Recommendation System
          </NavLogo>
        </NavbarContainer>
      </Nav>
    </>
  );
};

export default Navbar2;
