% abstract superclass for all geometric shapes
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-08-22
%
classdef (Abstract) shape

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected)

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = shape( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'shape:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create geometric shapes
            %--------------------------------------------------------------
            % repeat geometric shapes
            objects = repmat( objects, size );

        end % function objects = shape( size )

	end % methods (Access = protected)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        objects_out = center( shapes )

        %------------------------------------------------------------------
        % move
        %------------------------------------------------------------------
        shapes = move( shapes, centers )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        objects_out = discretize( shapes, options )

	end % methods (Abstract)

end % classdef (Abstract) shape
