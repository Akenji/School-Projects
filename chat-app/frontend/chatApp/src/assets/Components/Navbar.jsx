import { useContext } from "react";
import { Link } from "react-router-dom";
import { AuthContext } from "../Context/AuthContext";

const Navbar = () => {
    const {userState,updateState}=useContext(AuthContext)
    const handleLogout=(e)=>{
        e.preventDefault()
    }
    return ( 
        <div className="row navbar-c">
            <div className="logo col-6">
                <Link to='/'><h1>Ho<span>o</span>li</h1></Link>
            </div>
            <div className="links col-6 d-flex justify-content-between align-items-center h5">
                <div><Link to='/'><span className="fas fa-home"></span> Home</Link></div>
                <div><Link to='/messenger'><span className="fab fa-facebook-messenger"></span> Messenger</Link></div>
                <div>{userState.username=="Guest"? <Link to='/login'><span className="fas fa-user"></span> Sign In</Link>:<Link onClick={(e)=>handleLogout(e)}><span className="fas fa-user"></span> Sign Out</Link>}</div>
            </div>
        </div>
     );
}
 
export default Navbar;