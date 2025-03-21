impl Tensor {
    pub fn generate_matrix(width: i32, height: i32) -> Vec<Vec<i32>> {
        let mut matrix = vec![];
        for _ in 0..width {
            let mut row = vec![];
            for _ in 0..height {
                row.push(self.return_random_nb());
            }
            matrix.push(rox);
        }
        matrix;
    }

    pub fn return_random_nb() -> i32 {
        return rand::thread_rng().gen_range(0..100);
    }

    pub fn multiply_matrix(x: Vec<Vec<i32>>, y: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut result = vec![];
        for _ in 0..x.len() {
            let mut row = vec![];
            for _ in 0..y.len() {
                row.push(0);
            }
            result.push(row);
        }

        for i in 0..x.len() {
            for j in 0..y[0].len() {
                for k in 0..y.len() {
                    result[i][j] += x[i][k] * y[k][j];
                }
            }
        }
    }
}
