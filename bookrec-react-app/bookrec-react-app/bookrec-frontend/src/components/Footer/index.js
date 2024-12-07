import React from 'react'

import { FooterContainer, FooterWrap, SocialMedia, SocialMediaWrap, WebsiteRights } from './FooterElements';

const Footer = () => {
    return (
        <FooterContainer>
            <FooterWrap>
                <SocialMedia>
                    <SocialMediaWrap>
                    Book Recommendation System and Classification Using Machine Learning
                        <WebsiteRights>
                            Empirical Project © {new Date().getFullYear()}
                            &nbsp;All Rights Reserved
                        </WebsiteRights>
                    </SocialMediaWrap>
                </SocialMedia>
            </FooterWrap>
        </FooterContainer>
    );
}

export default Footer
