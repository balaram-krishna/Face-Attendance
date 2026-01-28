-- CREATE TABLE Students (
--     roll_no VARCHAR(20) PRIMARY KEY,
--     student_name VARCHAR(100),
--     class VARCHAR(10),
--     section VARCHAR(10),
--     image_path VARCHAR(255)
-- );

-- CREATE TABLE Attendance (
--     id INT IDENTITY PRIMARY KEY,
--     roll_no VARCHAR(20),
--     student_name VARCHAR(100),
--     class VARCHAR(10),
--     section VARCHAR(10),
--     subject VARCHAR(50),
--     faculty VARCHAR(50),
--     period VARCHAR(20),
--     date DATE,
--     time TIME,
--     status VARCHAR(10)
-- );
--  

select * from ICFAISMS.tblStudentCourseEnrollment;
