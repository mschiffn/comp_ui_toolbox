%
% superclass for all linear transforms (matrix processing)
%
% abstract superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2020-01-29
% modified: 2020-11-05
%
classdef (Abstract) linear_transform_matrix < linear_transforms.linear_transform

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = linear_transform_matrix( N_coefficients, N_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures nonempty positive integers for N_coefficients
            % superclass ensures nonempty positive integers for N_points
            % superclass ensures equal number of dimensions and sizes for N_coefficients and N_points

            %--------------------------------------------------------------
            % 2.) create linear transforms (matrix processing)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.linear_transform( N_coefficients, N_points )

        end % function objects = linear_transform_matrix( N_coefficients, N_points )

        %------------------------------------------------------------------
        % forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.linear_transform_matrix
            if ~isa( LTs, 'linear_transforms.linear_transform_matrix' )
                errorStruct.message = 'LTs must be linear_transforms.linear_transform_matrix!';
                errorStruct.identifier = 'forward_transform:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % ensure equal number of dimensions and sizes
            [ LTs, x ] = auxiliary.ensureEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate linear transforms
            for index_object = 1:numel( LTs )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % ensure equal numbers of points
                if size( x{ index_object }, 1 ) ~= LTs( index_object ).N_points
                    errorStruct.message = sprintf( 'x{ %d } must have %d rows!', index_object, LTs( index_object ).N_points );
                    errorStruct.identifier = 'forward_transform:InvalidNumberOfRows';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute forward transforms
                %----------------------------------------------------------
                % call forward transform for single matrix
                y{ index_object } = forward_transform_matrix( LTs( index_object ), x{ index_object } );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.linear_transform_matrix
            if ~isa( LTs, 'linear_transforms.linear_transform_matrix' )
                errorStruct.message = 'LTs must be linear_transforms.linear_transform_matrix!';
                errorStruct.identifier = 'adjoint_transform:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % ensure equal number of dimensions and sizes
            [ LTs, x ] = auxiliary.ensureEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute adjoint transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate linear transforms
            for index_object = 1:numel( LTs )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'adjoint_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % ensure equal numbers of coefficients
                if size( x{ index_object }, 1 ) ~= LTs( index_object ).N_coefficients
                    errorStruct.message = sprintf( 'x{ %d } must have %d rows!', index_object, LTs( index_object ).N_coefficients );
                    errorStruct.identifier = 'adjoint_transform:InvalidNumberOfRows';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute adjoint transforms
                %----------------------------------------------------------
                % call adjoint transform for single matrix
                y{ index_object } = adjoint_transform_matrix( LTs( index_object ), x{ index_object } );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % display coefficients
        %------------------------------------------------------------------
        function display_coefficients( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.linear_transform_matrix
            if ~isa( LTs, 'linear_transforms.linear_transform_matrix' )
                errorStruct.message = 'LTs must be linear_transforms.linear_transform_matrix!';
                errorStruct.identifier = 'display_coefficients:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % ensure equal number of dimensions and sizes
            [ LTs, x ] = auxiliary.ensureEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) display coefficients
            %--------------------------------------------------------------
            % iterate linear transforms
            for index_object = 1:numel( LTs )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'display_coefficients:NoNumericMatrix';
                    error( errorStruct );
                end

                % ensure equal numbers of coefficients
                if size( x{ index_object }, 1 ) ~= LTs( index_object ).N_coefficients
                    errorStruct.message = sprintf( 'x{ %d } must have %d rows!', index_object, LTs( index_object ).N_coefficients );
                    errorStruct.identifier = 'display_coefficients:InvalidNumberOfRows';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) display coefficients (single matrix)
                %----------------------------------------------------------
                display_coefficients_matrix( LTs( index_object ), x{ index_object } );

            end % for index_object = 1:numel( LTs )

        end % function display_coefficients( LTs, x )

        %------------------------------------------------------------------
        % relative RMSEs of the s largest expansion coefficients
        %------------------------------------------------------------------
        function rel_RMSEs = rel_RMSE( LTs, x, N_points_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two or three arguments
            narginchk( 2, 3 );

            % method forward_transform ensures class linear_transforms.linear_transform_matrix

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % ensure existence of nonempty N_points_s
            if nargin < 3 || isempty( N_points_s )
                N_points_s = 1500;
            end

            % ensure equal number of dimensions and sizes
            [ LTs, x, N_points_s ] = auxiliary.ensureEqualSize( LTs, x, N_points_s );

            %--------------------------------------------------------------
            % 2.) compute relative RMSEs of the s largest expansion coefficients
            %--------------------------------------------------------------
            % compute forward transforms
            y = forward_transform( LTs, x );

            % ensure cell array for y
            if ~iscell( y )
                y = { y };
            end

            % specify cell arrays
            rel_RMSEs = cell( size( LTs ) );
            axes_s = cell( size( LTs ) );

            % iterate linear transforms
            for index_object = 1:numel( LTs )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % method forward_transform ensures numeric matrix for x{ index_object }
                % method forward_transform ensures equal numbers of points in x{ index_object }

                % ensure valid number of evaluation points N_points_s( index_object )
                if N_points_s( index_object ) > LTs( index_object ).N_coefficients
                    errorStruct.message = sprintf( 'N_points_s( %d ) must be smaller or equal to number of coefficients %d!', index_object, LTs( index_object ).N_coefficients );
                    errorStruct.identifier = 'rel_RMSE:InvalidNumberOfRows';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute relative RMSEs of the s largest expansion coefficients
                %----------------------------------------------------------
                % call rel_RMSE for single matrix
                [ rel_RMSEs{ index_object }, axes_s{ index_object } ] = rel_RMSE_matrix( LTs( index_object ), x{ index_object }, y{ index_object }, N_points_s( index_object ) );

            end % for index_object = 1:numel( LTs )

            % create signal matrices
%             rel_RMSEs = processing.signal_matrix( math.sequence_increasing( axes_s ), rel_RMSEs );

        end % function rel_RMSEs = rel_RMSE( LTs, x, N_points_s )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single matrix)
        %------------------------------------------------------------------
        y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        display_coefficients_matrix( LT, x )

        %------------------------------------------------------------------
        % relative RMSEs of the s largest expansion coefficients (single matrix)
        %------------------------------------------------------------------
        [ rel_RMSEs, axis_s ] = rel_RMSE_matrix( LT, x, y, N_points_s )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) linear_transform_matrix < linear_transforms.linear_transform
