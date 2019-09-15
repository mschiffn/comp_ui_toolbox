%
% superclass for all images
%
% author: Martin F. Schiffner
% date: 2019-09-10
% modified: 2019-09-10
%
classdef image

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        grid ( 1, 1 ) math.grid
        samples ( :, : ) double

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = image( grids, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure class math.grid
            if ~isa( grids, 'math.grid' )
                errorStruct.message = 'grids must be math.grid!';
                errorStruct.identifier = 'image:NoGrids';
                error( errorStruct );
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % single grids / multiple samples
            if isscalar( grids ) && ~isscalar( samples )
                grids = repmat( grids, size( samples ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids, samples );

            %--------------------------------------------------------------
            % 2.) create images
            %--------------------------------------------------------------
            % repeat default image
            objects = repmat( objects, size( grids ) );

            % iterate images
            for index_object = 1:numel( objects )

                % ensure sample matrix
%                 if ~ismatrix( samples{ index_object } )
%                     errorStruct.message = sprintf( 'samples{ %d } must be a matrix!', index_object );
%                     errorStruct.identifier = 'image:NoMatrix';
%                     error( errorStruct );
%                 end

                % ensure correct sizes
%                 if abs( axes( index_object ) ) ~= size( samples{ index_object }, 1 )
%                     errorStruct.message = sprintf( 'Cardinality of axes( %d ) must match the size of samples{ %d } along the first dimension!', index_object, index_object );
%                     errorStruct.identifier = 'image:SizeMismatch';
%                     error( errorStruct );
%                 end

                % set independent properties
                objects( index_object ).grid = grids( index_object );
                objects( index_object ).samples = samples{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = image( axes, samples )

    end % methods

end % classdef image
