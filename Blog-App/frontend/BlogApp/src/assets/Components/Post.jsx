import { useContext } from "react"
import { AuthContext } from "../Context/AuthContext"
import { Link } from "react-router-dom"

const Post = ({id,title,image,body}) => {
    return ( 
        <div className="post w-100 shadow pb-2">
            <div className="post-img w-100">
                <img src={image} className="img-fluid" alt="" />
            </div>
            <div className="post-title p-2 h4 px-3 mb-0 pb-1">
                {title}
            </div>
            <div className="post-body p-2 px-3">
                {body.slice(0,50)}....
            </div>
            <div className="options p-2 px-3">
                <span className="me-2"><i className="fas fa-heart"></i> 8</span>
                <span className="ms-2"><i className="fas fa-comment"></i> 2</span>
            </div>
            {/* <Link to={"/details/"+id} className="p-2">See more</Link> */}
        </div>
     );
}
 
export default Post;