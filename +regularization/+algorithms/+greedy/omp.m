%
% superclass for all orthogonal matching pursuit (OMP) algorithms
%
% author: Martin F. Schiffner
% date: 2011-06-15
% modified: 2020-03-15
%
classdef omp < regularization.algorithms.greedy.greedy

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = omp( rel_RMSEs, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSEs
            % superclass ensures valid N_iterations_max

            %--------------------------------------------------------------
            % 2.) create OMP algorithms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.algorithms.greedy.greedy( rel_RMSEs, N_iterations_max );

        end % function objects = omp( rel_RMSEs, N_iterations_max )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_omp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.algorithms.greedy.omp
            if ~isa( algorithms_omp, 'regularization.algorithms.greedy.omp' )
                errorStruct.message = 'algorithms_omp must be regularization.algorithms.greedy.omp!';
                errorStruct.identifier = 'string:NoOMPRegularizationAlgorithms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "OMP (q = 0)"
            strs_out = repmat( "OMP (q = 0)", size( algorithms_omp ) );

        end % function strs_out = string( algorithms_omp )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % execute (implement execute_scalar method)
        %------------------------------------------------------------------
        function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_omp, op_A, y_m )
        % Recovers a sparse approximation of a specified vector y_m in
        % a specified dictionary op_A by the orthogonal matching pursuit (OMP)
        % (see [1, Sect. II.C.2]).
        %
        % INPUT:
        %	op_A = linear dictionary operator (matrix or function handle)
        %	y_m = vector of observations
        %
        % OUTPUT:
        %   theta_recon = recovered coefficients
        %   y_m_res = residual
        %   info = algorithm statistics (runtime)
        %
        % REFERENCES:
        %	[1] J. A. Tropp, "Greed is Good: Algorithmic Results for Sparse Approximation",
        %       IEEE Trans. Inf. Theory, Oct. 2004, Vol. 50, No. 10, pp. 2231-2242
        %       DOI: 10.1109/TIT.2004.834793
        %   [2] Y. C. Eldar, "Compressed Sensing: Algorithms and Applications"
        %
        % REMARKS:
        %   - Unlike the matching pursuit, OMP never selects the same atom twice.
        %	- Results of greedy algorithms might be suboptimal.

            % print status
            time_start = tic;
            auxiliary.print_header( "OMP v. 1.0" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class regularization.algorithms.algorithm (scalar) for algorithm_omp

            % ensure class regularization.algorithms.greedy.omp
            if ~isa( algorithm_omp, 'regularization.algorithms.greedy.omp' )
                errorStruct.message = 'algorithm_omp must be regularization.algorithms.greedy.omp!';
                errorStruct.identifier = 'omp:NoOptionsAlgorithmOMP';
                error( errorStruct );
            end

            % superclass ensures numeric matrix or function_handle for op_A
            % superclass ensures compatibility of y_m with op_A

            %--------------------------------------------------------------
            % 2.) 
            %--------------------------------------------------------------
            % size of the linear operator
            if isnumeric( op_A )
                size_A = size( op_A );
            else
                size_A = op_A( [], 0 );
            end
            N_observations = size_A( 1 );
            N_coefficients = size_A( 2 );

            % input vector and residual
            y_m = y_m( : );
            y_m_norm = norm( y_m, 2 );

            % maximum number of iterations
            N_iterations_max = min( algorithm_omp.N_iterations_max, N_observations );

            strings = { 'number of rows', 'number of columns'; 'y_m_norm', 'number of iterations' };
            N_chars_max = max( max( cellfun( @numel, strings ) ) );
            fprintf( ' %-20s: %7d %10s %-30s: %7d\n', 'number of rows', N_observations, '', 'number of columns', N_coefficients );
            fprintf( ' %-20s: %7.4f %10s %-30s: %7d\n', 'y_m_norm', y_m_norm, '', 'number of iterations', N_iterations_max );
            fprintf( ' %-20s: %7.4f\n', 'objective', algorithm_omp.rel_RMSEs );
            fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );

            %--------------------------------------------------------------
            % 3.) initialization
            %--------------------------------------------------------------
            % initialize reduced sensing matrix with selected columns
            A_sel = zeros( N_observations, N_iterations_max );

            % initialize atoms
            atoms = zeros( 1, N_iterations_max );

            % initialize residual and its norms
            y_m_res = y_m;
            y_m_res_norm_rel = zeros( 1, N_iterations_max );

            % initialize initial estimate
            theta_recon = zeros( N_coefficients, 1 );

            %--------------------------------------------------------------
            % 4.) recover transform coefficients
            %--------------------------------------------------------------
            % iterate relative RMSEs
            for index_rel_RMSE = 1:numel( algorithm_omp.rel_RMSEs )

% TODO: warm start for multiple rel_RMSEs

                fprintf( ' %-10s %-10s\n', 'Iter', 'rel. RMSE' );
                for k_iter = 1:N_iterations_max

                    %------------------------------------------------------
                    % 1.) select new atom
                    %------------------------------------------------------
                    % a.) apply adjoint operator
                    if isnumeric( op_A )
                        temp = op_A' * y_m_res;
                    else
                        temp = op_A( y_m_res, 2 );
                    end

                    % b.) select largest element and save its index
                    [ ~, atoms( k_iter ) ] = max( abs( temp( : ) ) );

                    % c.) extract column vector from sensing matrix
                    if isnumeric( op_A )
                        A_sel( :, k_iter ) = op_A( :, atoms( k_iter ) );
                    else
                        temp = zeros( N_coefficients, 1 );
                        temp( atoms( k_iter ) ) = 1;

                        A_sel( :, k_iter ) = op_A( temp, 1 );
                    end

                    %----------------------------------------------------------
                    % 2.) find optimal values of coefficients (least-squares approximation of original y_m)
                    %----------------------------------------------------------
                    coeff = ( A_sel( :, 1:k_iter )' * A_sel( :, 1:k_iter ) ) \ ( A_sel( :, 1:k_iter )' * y_m );
                    theta_recon( atoms( 1:k_iter ) ) = coeff;

                    %----------------------------------------------------------
                    % 3.) compute residual
                    %----------------------------------------------------------
                    y_approx = A_sel( :, 1:k_iter ) * coeff;
                    y_m_res = y_m - y_approx;
                    y_m_res_norm_rel( k_iter ) = norm( y_m_res, 2 ) / y_m_norm;

                    %----------------------------------------------------------
                    % graphical illustration
                    %----------------------------------------------------------
%                     axis = math.sequence_increasing_regular_quantized( 192, 573, physical_values.hertz( 11778.5630153119564056396484375 ) );
%                     y_m_tilde = signal( processing.signal_matrix( axis, physical_values.volt( reshape( y_m, [ 382, numel( y_m ) / 382 ] ) ) ), 200 );
%                     y_approx_tilde = signal( processing.signal_matrix( axis, physical_values.volt( reshape( y_approx, [ 382, numel( y_m ) / 382 ] ) ) ), 200 );
%                     y_m_res_tilde = signal( processing.signal_matrix( axis, physical_values.volt( reshape( y_m_res, [ 382, numel( y_m ) / 382 ] ) ) ), 200 );
%                     y_m_tilde_max = max( abs( y_m_tilde.samples(:) ) );
% 
%                     figure( k_iter );
%                     subplot( 2, 3, 1 );
%                     imagesc( 20*log10( abs( y_m_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
%                     subplot( 2, 3, 2 );
%                     imagesc( 20*log10( abs( y_approx_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
%                     subplot( 2, 3, 3 );
%                     imagesc( 20*log10( abs( y_m_res_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
%                     subplot( 2, 3, 4 );
%                     plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 32 ), 'b', ...
%                           (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 32 ), 'g', ...
%                           (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 32 ), 'r:' );
%                     subplot( 2, 3, 5 );
%                     plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 64 ), 'b', ...
%                           (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 64 ), 'g', ...
%                           (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 64 ), 'r:' );
%                     subplot( 2, 3, 6 );
%                     plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 96 ), 'b', ...
%                           (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 96 ), 'g', ...
%                           (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 96 ), 'r:' );
%                     colormap parula;

                    % print status
                    fprintf( ' %10d %10.4f\n', k_iter, y_m_res_norm_rel( k_iter ) );

                    %----------------------------------------------------------
                    % 4.) stopping criterion
                    %----------------------------------------------------------
                    if y_m_res_norm_rel( k_iter ) <= algorithm_omp.rel_RMSEs( index_rel_RMSE )
                        break;
                    end

                end % for k_iter = 1:N_iterations_max

            end % for index_rel_RMSE = 1:numel( algorithm_omp.rel_RMSEs )

            % truncate data structures to actual number of iterations
            atoms = atoms( 1:k_iter );
            y_m_res_norm_rel = y_m_res_norm_rel( 1:k_iter );

            % stop time measurement
            time_elapsed = toc( time_start );

            %--------------------------------------------------------------
            % 5.) return info structure
            %--------------------------------------------------------------
            info.size_A = size_A;
            info.atoms = atoms;
            info.y_m_res_norm_rel = y_m_res_norm_rel;
            info.time_total = time_elapsed;

        end % function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_omp, op_A, y_m )

	end % methods (Access = protected, Hidden)

end % classdef omp < regularization.algorithms.greedy.greedy
