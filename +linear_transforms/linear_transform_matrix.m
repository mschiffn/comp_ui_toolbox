%
% superclass for all linear transforms (matrix processing)
%
% abstract superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2020-01-29
% modified: 2020-02-06
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

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

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

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

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

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) linear_transform_matrix < linear_transforms.linear_transform
