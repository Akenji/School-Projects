const express=require('express')
const router=express.Router()
const passport=require('passport')
const jwt=require('jsonwebtoken')
const {register,login,logout,getPosts,addNewPost,deletePost,getUsers}=require('../controller/controller')
const multer=require('multer')

const isAuthenticated=(req,res,next)=>{
    const token=req.cookies.hooli
    if(token){
       jwt.verify(token,process.env.Secret,(err,decodedtoken)=>{
          if(err){
            console.log(err)
            // res.redirect('/')
            res.json({Error:"Couldn't verify token"})
            return
          }
          console.log(decodedtoken)
          next()
       })
    }else{
      res.json({Message:"Not logged in"})
    }
}

const storage=multer.diskStorage({
  destination:(req,file,cb)=>{
      cb(null,'frontend/chatApp/public/assets/img')
  },
  filename:(req,file,cb)=>{
      cb(null,file.originalname)
  }
})
const upload=multer({storage:storage})

router.get('/cookie',(req,res)=>{
    console.log("//////")
    console.log(req.cookies.authApp)
    console.log("//////")
    res.cookie("authApp",req.cookies.authApp,{maxAge:1000*3600*24})
    res.json({Message:req.cookies})
})
router.get("/home",(req,res)=>{
    res.send('<a href="/auth/google">Login with google</a>')
})
router.get('/auth/google',
  passport.authenticate('google', { scope: ['email','profile'] }));

router.get('/auth/google/callback', 
  passport.authenticate('google', { session:false,failureRedirect: '/login' }),
  function(req, res) {
    res.redirect('/');
  });

router.get("/auth",(req,res)=>{
    res.send("Reached")
})
router.get('/getUsers',isAuthenticated,getUsers)
router.post("/Register",register)
router.post("/Login",login)
router.get("/logout",logout)
router.get("/post/getPosts/:id",getPosts)
// router.post("/post/add/:id",upload.single('image'),addNewTask)
router.post("/post/add/:id",addNewPost)
router.post("/post/delete/:id",deletePost)


module.exports={router,isAuthenticated}