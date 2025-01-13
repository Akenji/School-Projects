import { useContext, useState } from "react";
import CreatePost from "./CreatePost";
import Post from "./Post";
import { AuthContext } from "../Context/AuthContext";
import { Link } from "react-router-dom";
import myImage from "../Images/Untitled5.png";

const Home = () => {
    const [showForm,setShowForm]=useState(false)
    const {userState,updateState}=useContext(AuthContext)
    return ( 
        <div className="home mt-4">
            <div><span className="h3 me-2">Welcome {userState.username}</span>
            {userState.username!=="Guest" && <button className="btn btn-custom btn-lg my-2" onClick={()=>setShowForm(!showForm)}>Create Post</button>}
            {userState.username == "Guest" && <Link to='/login'><button className="btn btn-custom btn-lg my-2">Sign In</button></Link>}
            </div> 
            
            <div className="hero-image" style={{display: 'flex', alignItems:'center', justifyContent:'center', gap:'2rem'}}>
        <span style={{color:'darkblue', fontSize:'20px'}}>Connect with friends <br />and the world around you <br />on Chit-Chat</span>
        <img src={myImage} alt="my hero image" style={{ border:'1px solid blue'}}/>
        </div>


            {showForm && <CreatePost/>}
            {userState.username!=="Guest" &&
            (userState.posts.length>0
            
            ?
            <div className="row justify-content-between px-1">
                {
                    userState.posts.map((post,e)=>(
                        <Link to={"/details/"+post.id} className="col-md-5 col my-3" key={e}>
                        <Post id={post.id} title={post.title} image={"/assets/img/"+post.image} body={post.body} />
                        </Link>
                    ))
                }
            </div>
            :
            <p className="h4">No posts yet</p>
            )
            }
        </div>
     );
}
 
export default Home;