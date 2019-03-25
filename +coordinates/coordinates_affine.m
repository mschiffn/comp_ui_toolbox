%
% superclass for all affine coordinates
%
% author: Martin F. Schiffner
% date: 2019-03-22
% modified: 2019-03-25
%
classdef coordinates_affine < coordinates.coordinates

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        components ( 1, : ) physical_values.length

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coordinates_affine( components )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                components = physical_values.length( zeros( 1, 2 ) );
            end

            % convert matrix to cell array
            if ismatrix( components )
                components = mat2cell( components, ones( size( components, 1 ), 1 ) );
            end

            % ensure cell array for components
            if ~iscell( components )
                components = { components };
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@coordinates.coordinates( size( components ) );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )
                objects( index_object ).components = components{ index_object };
            end

        end % function objects = coordinates_affine( components )

        %------------------------------------------------------------------
        % addition (overload plus function)
        %------------------------------------------------------------------
        function results = plus( objects_1, objects_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal classes
            if ~strcmp( class( objects_1 ), class( objects_2 ) )
                errorStruct.message     = 'Both arguments must have the same class!';
                errorStruct.identifier	= 'plus:ClassMismatch';
                error( errorStruct );
            end

            % multiple objects_1 / single objects_2
            if ~isscalar( objects_1 ) && isscalar( objects_2 )
                objects_2 = repmat( objects_2, size( objects_1 ) );
            end

            % single objects_1 / multiple objects_2
            if isscalar( objects_1 ) && ~isscalar( objects_2 )
                objects_1 = repmat( objects_1, size( objects_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 );

            %--------------------------------------------------------------
            % 2.) perform addition
            %--------------------------------------------------------------
            % create results of the same class
            results = objects_1;

            % add the affine coordinates
            % TODO: very slow, use matrix operations
            for index_object = 1:numel( objects_1 )
                results( index_object ).components = objects_1( index_object ).components + objects_2( index_object ).components;
            end

        end % function results = plus( objects_1, objects_2 )

    end % methods

end % classdef coordinates_affine < coordinates.coordinates
