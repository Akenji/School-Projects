<?php
echo "inside" ;
   $username = $_POST['username'];
  $email = $_POST['email'];
  $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];
 $conn = new mysqli('localhost', 'root','','User');
 if($conn->connect_error){
    die('connection failed :' .$conn->connect_error );
 }
 else{
    $stmt = $conn->prepare("insert into medic_db(username, email, password , confirm password) 
    values(?,?,?,?)");
    $stmt->bind_param("ssss", $username, $email, $password, $confirm_password);
    $stmt->execute();
    echo "registration succesfully";
    $stmt->close();
    $stmt->close();

 }
?>