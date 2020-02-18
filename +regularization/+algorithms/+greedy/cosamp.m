%
% superclass for all compressive sampling matching pursuit (CoSaMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-22
% modified: 2020-02-16
%
classdef cosamp < regularization.algorithms.greedy.greedy

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sparsity ( 1, 1 ) double { mustBePositive, mustBeInteger } = 10	% sparsity level

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = cosamp( rel_RMSE, N_iterations_max, sparsity )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max
            % property validation functions ensure valid sparsity

            % multiple rel_RMSE / single q
            if ~isscalar( rel_RMSE ) && isscalar( sparsity )
                sparsity = repmat( sparsity, size( rel_RMSE ) );
            end

            %--------------------------------------------------------------
            % 2.) create CoSaMP options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.algorithms.greedy.greedy( rel_RMSE, N_iterations_max );

            % iterate CoSaMP options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).sparsity = sparsity( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = cosamp( rel_RMSE, N_iterations_max )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( options_cosamp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.algorithms.greedy.cosamp
            if ~isa( options_cosamp, 'regularization.algorithms.greedy.cosamp' )
                errorStruct.message = 'options_cosamp must be regularization.algorithms.greedy.cosamp!';
                errorStruct.identifier = 'string:NoOptionsCoSaMP';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( options_cosamp ) );

            % iterate SPGL1 options
            for index_object = 1:numel( options_cosamp )

                strs_out( index_object ) = sprintf( "%s (s = %d)", 'CoSaMP', options_cosamp( index_object ).sparsity );

            end % for index_object = 1:numel( options_cosamp )

        end % function strs_out = string( options_cosamp )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % execute (implement execute_scalar method)
        %------------------------------------------------------------------
        function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_cosamp, op_A, y_m )
        %
        % compressive sampling matching pursuit (CoSaMP) to recover a sparse coefficient vector
        % (cf. [1, Algorithm 1])
        %
        % INPUT:
        %	op_A = linear dictionary operator (matrix or function handle)
        %	y_m = vector of observations
        %	algorithm_cosamp = regularization.algorithms.greedy.cosamp
        %
        % OUTPUT:
        %   theta_recon = recovered coefficients
        %   y_m_res = residual
        %   info = algorithm statistics (runtime)
        %
        % REFERENCES:
        %	[1] D. Needell, J. A. Tropp, "CoSaMP: Iterative signal recovery from incomplete and inaccurate samples,"
        %       Appl. Comput. Harmon. A., 2009, vol. 26, no. 3, pp. 301-321, DOI: 10.1016/j.acha.2008.07.002
        %
        % REMARKS:
        %
        % author: Martin F. Schiffner
        % date: 2013-05-13
        % modified: 2020-02-12

            % internal constant
            N_atoms_max = 1024;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );
            fprintf( ' %s (%s)\n', 'CoSaMP v. 0.1', str_date_time );
            fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );

            %--------------------------------------------------------------
            % 1.) check arguments
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

            % ensure class regularization.algorithms.greedy.cosamp
            if ~isa( algorithm_cosamp, 'regularization.algorithms.greedy.cosamp' )
                errorStruct.message = 'algorithm_cosamp must be regularization.algorithms.greedy.cosamp!';
                errorStruct.identifier = 'omp:NoOptionsAlgorithmCoSaMP';
                error( errorStruct );
            end

            % maximum number of iterations
            N_iterations_max = min( algorithm_cosamp.N_iterations_max, N_observations );

            %--------------------------------------------------------------
            % 2.) initialization
            %--------------------------------------------------------------
            % initialize reduced sensing matrix with selected columns
            A = [];

            % initialize atoms
            atoms = [];

            % initialize residual and its norms
            y_m_res = y_m;
            y_m_res_norm_rel = zeros( 1, N_iterations_max );

            %--------------------------------------------------------------
            % 3.) recover transform coefficients
            %--------------------------------------------------------------
            for k_iter = 1:N_iterations_max

                %----------------------------------------------------------
                % 1.) select atoms
                %----------------------------------------------------------
                % a.) apply adjoint operator (form signal proxy)
                if isnumeric( op_A )
                    temp = op_A' * y_m_res;
                else
                    temp = op_A( y_m_res, 2 );
                end

                % b.) identify the 2s largest components
                [ ~, atoms_identified ] = sort( abs( temp( : ) ), 'descend' );
                atoms_identified = atoms_identified( 1:2*algorithm_cosamp.sparsity );

                % c.) merge supports
                atoms_old = atoms;
                [ atoms, indices_old, indices_new ] = union( atoms_old, atoms_identified );
                atoms_old = atoms_old( indices_old );
                atoms_new = atoms_identified( indices_new );
                N_atoms_new = numel( atoms_new );

                % select column vectors of old atoms
                A_old = A( :, indices_old );

                %----------------------------------------------------------
                % 2.) extract novel column vectors from sensing matrix
                %----------------------------------------------------------
                % a) partition new atoms into batches to save memory
                N_batches = ceil( N_atoms_new / N_atoms_max );
                N_atoms_last = N_atoms_new - ( N_batches - 1 ) * N_atoms_max;
                indices = mat2cell( (1:N_atoms_new), 1, [ N_atoms_max * ones( 1, N_batches - 1 ), N_atoms_last ] );

                % b) initialize new sensing matrix
                A_new = zeros( N_observations, N_atoms_new );

                % c) iterate batches
                for index_batch = 1:N_batches

                    % a) indices of coefficients
                    indices_theta = ( 0:( numel( indices{ index_batch } ) - 1 ) ) * N_coefficients + atoms_new( indices{ index_batch } )';

                    % b) initialize coefficient vectors
                    theta = zeros( N_coefficients, numel( indices{ index_batch } ) );
                    theta( indices_theta ) = 1;

                    % c) 
                    A_new( :, indices{ index_batch } ) = op_A( theta, 1 );

                end % for index_batch = 1:N_batches

                %----------------------------------------------------------
                % 3.) update sensing matrix
                %----------------------------------------------------------
                % construct novel dictionary matrix
                A = zeros( N_observations, numel( atoms ) );
                if k_iter ~= 1
                    [ ~, index_in_atoms ] = ismember( atoms_old, atoms );
                    A( :, index_in_atoms ) = A_old;
                end
                [ ~, index_in_atoms ] = ismember( atoms_new, atoms );
                A( :, index_in_atoms ) = A_new;

                %----------------------------------------------------------
                % 4.) find optimal values of coefficients (least-squares approximation of original y_m)
                %----------------------------------------------------------
                % find optimal values of coefficients (least-squares approximation)
                %coeff = (A' * A) \ (A' * y_m);
                coeff = A \ y_m;

                % prune atoms (retain s largest entries)
                [ ~, indices ] = sort( abs( coeff ), 'descend' );
                atoms = atoms( indices( 1:algorithm_cosamp.sparsity ) );
                A = A( :, indices( 1:algorithm_cosamp.sparsity ) );

                % recovered coefficient vector
                theta_recon = zeros( N_coefficients, 1 );
                theta_recon( atoms ) = coeff( indices( 1:algorithm_cosamp.sparsity ) );

                %----------------------------------------------------------
                % 5.) compute residual
                %----------------------------------------------------------
                y_approx = A * theta_recon( atoms );
%                 y_approx_old = op_A( theta_recon, 1 );
%                 fprintf( ' %2.4f\n', norm( y_approx - y_approx_old ) / norm( y_approx_old ) );
                y_m_res = y_m - y_approx;
                y_m_res_norm_rel( k_iter ) = norm( y_m_res, 2 ) / y_m_norm;

                % print status
                fprintf( ' %10d %10.4f\n', k_iter, y_m_res_norm_rel( k_iter ) );

                %----------------------------------------------------------
                % 6.) stopping criterion
                %----------------------------------------------------------
                if y_m_res_norm_rel( k_iter ) <= algorithm_cosamp.rel_RMSE
                    break;
                end

            end % for k_iter = 1:N_iterations_max

            % stop time measurement
            time_elapsed = toc( time_start );

            %--------------------------------------------------------------
            % 4.) return info structure
            %--------------------------------------------------------------
            info.size_A = size_A;
            info.atoms = atoms;
            info.y_m_res_norm_rel = y_m_res_norm_rel;
            info.time_total = time_elapsed;

        end % function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_cosamp, op_A, y_m )

	end % methods (Access = protected, Hidden)

end % classdef cosamp < regularization.algorithms.greedy.greedy
