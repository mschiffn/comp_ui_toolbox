%
% superclass for all inactive reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-16
%
classdef reweighting_off < regularization.options.reweighting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = reweighting_off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.reweighting( size );

        end % function objects = reweighting_off( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( reweightings_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.reweighting_off
            if ~isa( reweightings_off, 'regularization.options.reweighting_off' )
                errorStruct.message = 'reweightings_off must be regularization.options.reweighting_off!';
                errorStruct.identifier = 'string:NoOptionsReweightingOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( reweightings_off ) );

        end % function strs_out = string( reweightings_off )

	end % methods

end % classdef reweighting_off < regularization.options.reweighting
