import React, { useState } from "react";
import Navbar2 from "../../components/Navbar/navbar2";
import {
  FeaturesSec
} from "./Features.elements";
import {
  Card,
  CardContent,
  Typography,
  CardActionArea,
  Grid,
  TextField,
  Button,
  CircularProgress,
  Container,
} from "@mui/material";
import BookClassificationServices from "../../services/BookClassificationServices";
import "./Features.css"; // Add your custom styles here

const SimilarBooks = () => {
  const [bookTitle, setBookTitle] = useState('');
  const [submittedTitle, setSubmittedTitle] = useState('');
  const [similarBooks, setSimilarBooks] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmittedTitle(bookTitle);
    setLoading(true);

    BookClassificationServices.getSimilarBooks(bookTitle)
      .then(response => {
        console.log("Similar Books:", response);
        setSimilarBooks(response.data);
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Failed to find similar books.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <>
      <Navbar2 />
      <FeaturesSec>
        <Container maxWidth="md" className="find-books-container">
          <Typography variant="h4" className="section-title" style={{ textAlign: 'center', color: '#fff', marginBottom: '20px' }}>
            Find Similar Books
          </Typography>
          <form onSubmit={handleSubmit} className="form-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h6" style={{ color: '#fff', marginBottom: '8px', alignSelf: 'flex-start' }}>
              Enter Book Title
            </Typography>
            <TextField
              variant="outlined"
              value={bookTitle}
              onChange={(e) => setBookTitle(e.target.value)}
              placeholder="e.g., Charlotte"
              fullWidth
              margin="normal"
              InputLabelProps={{
                shrink: true,
              }}
              style={{ marginBottom: '20px', backgroundColor: '#fff', borderRadius: '8px' }}
            />

            <Button
              type="submit"
              variant="contained"
              color="primary"
              className="submit-button"
              fullWidth
              style={{ height: '48px', borderRadius: '8px', marginBottom: '20px' }}
            >
              {loading ? <CircularProgress size={24} style={{ color: '#fff' }} /> : "Find Similar Books"}
            </Button>
          </form>

          {submittedTitle && !loading && (
            <Typography variant="h6" className="result-heading" style={{ marginTop: '20px', textAlign: 'center', color: '#fff' }}>
              Similar Books for: <strong>{submittedTitle}</strong>
            </Typography>
          )}

          {similarBooks.length > 0 && (
            <Grid container spacing={3} justifyContent="center" style={{ marginTop: '20px' }}>
              {similarBooks.map((book, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card className="book-card" style={{ borderRadius: '12px', backgroundColor: '#f5f5f5' }}>
                    <CardActionArea>
                      <CardContent>
                        <Typography variant="h6" component="div" className="book-title" style={{ fontWeight: 'bold' }}>
                          {book["Book-Title"]}
                        </Typography>
                        <Typography variant="subtitle1" color="textSecondary" className="book-author" style={{ marginTop: '8px' }}>
                          by {book["Book-Author"]}
                        </Typography>
                        <Typography variant="body2" color="textSecondary" className="book-year" style={{ marginTop: '4px' }}>
                          Published: {book["Year-Of-Publication"]}
                        </Typography>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}

          {!loading && submittedTitle && similarBooks.length === 0 && (
            <Typography variant="body1" color="textSecondary" className="no-results" style={{ marginTop: '20px', textAlign: 'center' }}>
              No similar books found for this title.
            </Typography>
          )}
        </Container>
      </FeaturesSec>
    </>
  );
};

export default SimilarBooks;
