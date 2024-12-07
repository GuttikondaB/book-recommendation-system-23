import React, { useState } from "react";
import Navbar2 from "../../components/Navbar/navbar2";
import "./Features.css"; // Add any custom styles here for the card design
import BookClassificationServices from "../../services/BookClassificationServices";
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
} from "@mui/material";

const ClusterRecommendations = () => {
  const [userId, setUserId] = useState('');
  const [submittedUserId, setSubmittedUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmittedUserId(userId);
    setLoading(true);

    BookClassificationServices.getClusterRecommendations(userId)
      .then(response => {
        console.log("Cluster-based Recommendations:", response);
        setRecommendations(response.data);
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Failed to fetch recommendations.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <>
      <Navbar2 />
      <FeaturesSec>
        <h1>Cluster-Based Book Recommendations</h1>
        <div className="recommendation-container">
          <form onSubmit={handleSubmit} className="form-container">
            <Typography variant="h6" style={{ color: '#fff', marginBottom: '8px' }}>
              Enter User ID
            </Typography>
            <TextField
              variant="outlined"
              type="number"
              className="user-id-input"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., 12345"
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
              style={{ marginBottom: '20px', backgroundColor: '#fff', borderRadius: '4px' }}
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              className="recommend-button"
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : "Get Recommendations"}
            </Button>
          </form>
          {submittedUserId && (
            <Typography variant="h6" className="result-heading">
              Recommendations for User ID: <strong>{submittedUserId}</strong>
            </Typography>
          )}
          {recommendations.length > 0 ? (
            <div className="recommendation-list">
              <Grid container spacing={3} justifyContent="center">
                {recommendations.map((book, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card className="book-card">
                      <CardActionArea>
                        <CardContent>
                          <Typography variant="h6" component="div" className="book-title">
                            {book["Book-Title"]}
                          </Typography>
                          <Typography variant="subtitle1" color="textSecondary" className="book-author">
                            by {book["Book-Author"]}
                          </Typography>
                          <Typography variant="body2" color="textSecondary" className="book-year">
                            Year of Publication: {book["Year-Of-Publication"]}
                          </Typography>
                        </CardContent>
                      </CardActionArea>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </div>
          ) : (
            submittedUserId && !loading && (
              <Typography variant="body1" color="textSecondary" className="no-results">
                No recommendations available for this user ID.
              </Typography>
            )
          )}
        </div>
      </FeaturesSec>
    </>
  );
};

export default ClusterRecommendations;